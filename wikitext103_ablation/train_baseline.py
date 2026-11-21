import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
import os
import numpy as np
import time
import random
import sys
import datetime
import csv
import argparse
import logging

# Enable TF32 for massive speedups on Ampere/Ada GPUs (RTX 30/40 series)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# =====================================================================
# 1. Configuration & CLI Setup
# =====================================================================
parser = argparse.ArgumentParser(description="TMLR Paper Baseline Reproducibility Run")
parser.add_argument(
    '--baseline_mode',
    type=str,
    default='standard',
    choices=['standard', 'hybrid_peri_ln', 'relu_squared'],
    help="Select architecture baseline: 'standard' (Baseline 3), 'hybrid_peri_ln' (Baseline 1), or 'relu_squared' (Baseline 2)."
)
args = parser.parse_args()

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

class Config:
    # --- ARCHITECTURE TOGGLES ---
    attn_mode = 'dpa' 
    dpaloss = True 
    trn_shft_loss = False
    use_rope = True       # Toggles between RoPE and Absolute Positional Embeddings

    # --- Matched Topology ---
    seq_len = 512          
    embed_dim = 256        
    num_layers = 15        
    num_heads = 4          
    ffn_dim = 1024         
    
    # --- Matched Training Dynamics ---
    lr = 0.002 
    epochs = 100            
    batch_size = 32      
    dropout = 0.1          
    max_norm = 10.0      
    
    preNorm = False 
    postNorm = True        # Moves RMSNorm to the output of SA/FFN
    use_qk_norm = True

    # --- No-Softmax Loss Settings ---
    logitsclamp = 50.0
    usesharp = True
    sharppen = 0.03        # Tuned sweet spot

    start_decay = 0
    vocab_size = 50257     # Dynamically updated by tiktoken

# Apply CLI-selected baseline configurations
runbaseline = False
runGemRSquared = False

if args.baseline_mode == 'standard':
    runbaseline = True
    Config.attn_mode = 'standard'
    Config.dpaloss = False 
    Config.preNorm = True 
elif args.baseline_mode == 'relu_squared':
    runGemRSquared = True
    Config.attn_mode = 'dpa'
    Config.dpaloss = False 
    Config.preNorm = True 
elif args.baseline_mode == 'hybrid_peri_ln':
    # Baseline 1: Hybrid Peri-LN with DPM Loss
    Config.attn_mode = 'standard'
    Config.dpaloss = True
    Config.preNorm = False
    Config.postNorm = True

# =========================================================
# STANDARD LOGGING SETUP
# =========================================================
model_tag = f"Model_A_{Config.attn_mode.upper()}_{'RoPE' if Config.use_rope else 'AbsPos'}_{Config.num_layers}L_{args.baseline_mode.upper()}"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
log_path = os.path.join(log_dir, f"{Config.num_layers}_LAYER_{model_tag}_RunLog_{timestamp}.txt")
csv_log_path = os.path.join(log_dir, f"{Config.num_layers}_LAYER_{model_tag}_GradProfile_{timestamp}.csv")

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"[SYSTEM] Standard Logging Initialized. Saving terminal output to: {log_path}")

# Device configuration (MPS for Mac, CUDA for NVIDIA, CPU fallback)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

logger.info(f"===========================================================")
logger.info(f"  {model_tag} - {device.type.upper()}. {Config.num_layers} LAYERS")
logger.info(f"===========================================================\n")

logger.info("[------------------- RUN CONFIGURATION -------------------]")
for key, value in vars(Config).items():
    if not key.startswith('__') and not callable(value):
        logger.info(f"  {key:<22}: {value}")
logger.info(f"baseline_mode = {args.baseline_mode} (runbaseline={runbaseline}, runGemRSquared={runGemRSquared})")
logger.info("[---------------------------------------------------------]\n")

# =====================================================================
# 2. Data Loading
# =====================================================================
try:
    enc = tiktoken.get_encoding("gpt2")
    Config.vocab_size = enc.n_vocab
    train_bin_path = "wiki_train.bin"
    val_bin_path = "wiki_val.bin"

    raw_train = np.memmap(train_bin_path, dtype=np.uint16, mode='r').astype(np.int64)
    raw_val = np.memmap(val_bin_path, dtype=np.uint16, mode='r').astype(np.int64)
    
    train_data = torch.from_numpy(raw_train).to(device)
    val_data = torch.from_numpy(raw_val).to(device)
except FileNotFoundError:
    logger.error("[CRITICAL] Data files not found. Please ensure wiki_train.bin and wiki_val.bin exist in the root directory.")
    sys.exit(1)

def init_csv_logger(filepath, total_layers):
    headers = ['Step'] + [f'L{i}_Grad' for i in range(total_layers)]
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    logger.info(f"[SYSTEM] CSV Logger Initialized: {filepath}")

def get_layer_grad_norm(layer):
    total_norm = 0.0
    for p in layer.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# =====================================================================
# 3. Universal Attention
# =====================================================================
class UniversalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.embed_dim // cfg.num_heads
        
        self.W_q = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.W_k = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.W_v = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.W_o = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        
        if cfg.use_rope:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            t = torch.arange(cfg.seq_len)
            freqs = torch.outer(t, inv_freq)
            emb = torch.repeat_interleave(freqs, 2, dim=-1)
            self.register_buffer('rope_cos', emb.cos()[None, None, :, :])
            self.register_buffer('rope_sin', emb.sin()[None, None, :, :])

        if cfg.attn_mode == 'dpa' or getattr(cfg, 'use_qk_norm', False):
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        if cfg.attn_mode == 'dpa':
            self.poly_a = nn.Parameter(torch.tensor(1.0))
            self.poly_b = nn.Parameter(torch.tensor(1.0))
            self.poly_z_exp = nn.Parameter(torch.tensor(1.5)) 
            c_mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).view(1, 1, cfg.seq_len, cfg.seq_len)
            self.register_buffer('causal_mask', c_mask)

    def forward(self, x):
        B, T, C = x.size()
        
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.cfg.attn_mode == 'dpa' or getattr(self.cfg, 'use_qk_norm', False):
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        
        if self.cfg.use_rope:
            cos = self.rope_cos[:, :, :T, :]
            sin = self.rope_sin[:, :, :T, :]
            def rotate_interleaved(tensor):
                x1 = tensor[..., 0::2] 
                x2 = tensor[..., 1::2] 
                return torch.stack((-x2, x1), dim=-1).reshape(tensor.shape)
            Q = (Q * cos) + (rotate_interleaved(Q) * sin)
            K = (K * cos) + (rotate_interleaved(K) * sin)
        
        if self.cfg.attn_mode == 'dpa':
            causal_mask = self.causal_mask[:, :, :T, :T] 
            thermal_div = max(8.0, math.sqrt(self.head_dim))
            scores = torch.matmul(Q, K.transpose(-2, -1)) / thermal_div
            if not self.training: self.live_raw_scores = scores.detach()
            scores = scores.masked_fill(causal_mask == 0, 0.0)
            r_attn = F.relu(scores)
            
            if runGemRSquared:
                a = 1.0
                b = 0.0
                r_safe = r_attn
            else:
                b = F.softplus(self.poly_b)
                a = F.softplus(self.poly_a)
                r_safe = r_attn + 1e-6
            z_exp = 2.0 + F.softplus(self.poly_z_exp)
            attn_raw = (a * (r_safe ** 2.0)) + (b * (r_safe ** z_exp))
            attn_sum = attn_raw.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            attn = attn_raw / attn_sum
            attn_out = torch.matmul(attn, V)
        else:
            attn_out = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.cfg.dropout if self.training else 0.0, is_causal=True)
            
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(attn_out)

# =====================================================================
# 4. Standard Transformer Block & Model
# =====================================================================
class StandardTransformerBlock(nn.Module):
    def __init__(self, cfg): 
        super().__init__()
        self.cfg = cfg
        
        self.ln1 = nn.RMSNorm(cfg.embed_dim)
        self.ln2 = nn.RMSNorm(cfg.embed_dim)
        
        if getattr(cfg, 'postNorm', False):
            self.ln1_post = nn.RMSNorm(cfg.embed_dim)
            self.ln2_post = nn.RMSNorm(cfg.embed_dim)
            
        self.attn = UniversalAttention(cfg)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout)
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        norm_x = self.ln1(x) if getattr(self.cfg, 'preNorm', True) else x
        attn_out = self.attn(norm_x)
        
        if getattr(self.cfg, 'postNorm', False):
            attn_out = self.ln1_post(attn_out)
            
        x = x + self.dropout(attn_out)
        
        norm_x2 = self.ln2(x) if getattr(self.cfg, 'preNorm', True) else x
        ffn_out = self.ffn(norm_x2)

        if getattr(self.cfg, 'postNorm', False):
            ffn_out = self.ln2_post(ffn_out)
            
        x = x + self.dropout(ffn_out)
        return x

class StandardTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        
        if not cfg.use_rope:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.embed_dim)
        
        self.layers = nn.ModuleList([
            StandardTransformerBlock(cfg) 
            for _ in range(cfg.num_layers)                
        ])
        
        self.ln_f = nn.RMSNorm(cfg.embed_dim)
        self.vocab_proj = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.vocab_proj.weight = self.token_embed.weight
        self.dropout = nn.Dropout(cfg.dropout)

        self.loss_poly_a = nn.Parameter(torch.tensor(1.0))
        self.loss_poly_b = nn.Parameter(torch.tensor(1.0))
        self.loss_poly_z_exp = nn.Parameter(torch.tensor(1.5))
        self.loss_bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, T = x.size()
        
        embed = self.token_embed(x)
        if not self.cfg.use_rope:
            positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
            embed = embed + self.pos_embed(positions)
            
        x = self.dropout(embed)
                
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        return self.vocab_proj(x), x

# =====================================================================
# 5. Native No-Softmax Loss (Fused NSM Loss)
# =====================================================================
@torch.compile(mode="default")
def fused_nsm_loss(logits, targets, epoch=1, poly_a=None, poly_b=None, poly_z_exp=None, loss_bias=None, use_sharpness=True):
    logits_f32 = logits.float().clamp(min=-Config.logitsclamp, max=Config.logitsclamp)
    
    if Config.trn_shft_loss and loss_bias is not None:
        logits_f32 = logits_f32 + loss_bias

    r = F.relu(logits_f32)
    
    a = F.softplus(poly_a)
    b = F.softplus(poly_b)
    z_exp = 2.0 + F.softplus(poly_z_exp)
    
    r_safe = r + 1e-6
    activations = (a * (r_safe ** 2.0)) + (b * (r_safe ** z_exp))
    
    prob_sum = activations.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    probs = activations / prob_sum

    log_probs = torch.log(probs.clamp(min=1e-9))
    nll = F.nll_loss(log_probs.view(-1, Config.vocab_size), targets.view(-1))

    if Config.usesharp and use_sharpness:
        penalty_weight = min(Config.sharppen, (epoch / 10.0) * Config.sharppen) if epoch <= 10 else Config.sharppen
        sharpness_penalty = torch.sum(probs * log_probs, dim=-1).mean()
        return nll, nll - (penalty_weight * sharpness_penalty)
    
    return nll, nll

def generate_sample_text(
    model, enc, prompt_text="The universe is continuously expanding, and", max_new_tokens=60
):
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    base_model.eval()
    prompt_tokens = enc.encode(prompt_text)
    context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_crop = context[:, -Config.seq_len:]
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits_out, _ = base_model(context_crop)
                logits = logits_out[0, -1, :].float().clamp(min=-Config.logitsclamp, max=Config.logitsclamp)
                
                if Config.dpaloss:
                    r_gen = F.relu(logits)
                    a = F.softplus(base_model.loss_poly_a)
                    b = F.softplus(base_model.loss_poly_b)
                    z_exp = 2.0 + F.softplus(base_model.loss_poly_z_exp)
                    r_safe = r_gen + 1e-6
                    activations = (a * (r_safe ** 2.0)) + (b * (r_safe ** z_exp))
                    
                    v, _ = torch.topk(activations, min(50, activations.size(-1)))
                    activations[activations < v[-1]] = 0.0
                    probs = activations / activations.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                else:
                    v, _ = torch.topk(logits, min(50, logits.size(-1)))
                    logits[logits < v[-1]] = float('-inf')
                    probs = F.softmax(logits.float(), dim=-1)
                    
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, next_token.unsqueeze(0)), dim=1)
                
    return enc.decode(context[0].tolist())

# =====================================================================
# 6. Multi-Seed Automated Training Loop
# =====================================================================
def get_batch(data):
    ix = torch.randint(len(data) - Config.seq_len - 1, (Config.batch_size,), device=device)
    offsets = torch.arange(Config.seq_len, device=device)
    idx = ix.unsqueeze(1) + offsets
    return data[idx], data[idx + 1]

steps_per_epoch = 1000  
seed_list = [42, 105, 555, 665]

val_loss_history = {seed: [] for seed in seed_list}
generated_samples = {}

init_csv_logger(csv_log_path, total_layers=Config.num_layers)

for seed in seed_list:
    logger.info(f"\n{'='*60}")
    logger.info(f" STARTING RUN FOR SEED: {seed}")
    logger.info(f"{'='*60}")
    val_low = float('inf')
    
    # Guaranteed Hardware Determinism Setup
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model = StandardTransformer(Config).to(device)

    logger.info("\n[SYSTEM] Fusing graph with torch.compile(). This will take 1-3 minutes...")
    model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[ARCHITECTURE] Total Trainable Parameters: {total_params:,}")

    core_params = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and 'token_embed' not in n and 'pos_embed' not in n and 'vocab_proj' not in n
    )
    logger.info(f"[ARCHITECTURE] Core Backbone Parameters: {core_params:,}")

    if runGemRSquared:
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if param.ndim < 2 or 'poly' in name or 'bias' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': 0.0, 'is_decay': True}, 
            {'params': no_decay_params, 'weight_decay': 0.0, 'is_decay': False}
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=Config.lr, fused=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=1e-4, fused=True)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.lr,                      
        epochs=Config.epochs,                
        steps_per_epoch=steps_per_epoch,    
        pct_start=0.05,                      
        anneal_strategy='cos',  
        div_factor=5.0,       
        final_div_factor=10.0 
    )

    for epoch in range(1, Config.epochs + 1):
        if runGemRSquared:
            current_wd = 1e-4 if epoch > Config.start_decay else 0.0
            for param_group in optimizer.param_groups:
                if param_group.get('is_decay', False):
                    param_group['weight_decay'] = current_wd

        model.train()
        epoch_loss = 0.0
        epoch_raw = 0.0
        epoch_start_time = time.time()
        
        # Buffered CSV I/O
        csv_buffer = []

        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            
            X_batch, Y_batch = get_batch(train_data)
        
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16): 
                logits, _ = model(X_batch)
                
                if Config.dpaloss:
                    base_model_ref = model._orig_mod if hasattr(model, '_orig_mod') else model
                    raw_loss, loss = fused_nsm_loss(
                        logits, Y_batch, epoch=epoch, 
                        poly_a=base_model_ref.loss_poly_a, 
                        poly_b=base_model_ref.loss_poly_b, 
                        poly_z_exp=base_model_ref.loss_poly_z_exp,
                        loss_bias=base_model_ref.loss_bias,
                        use_sharpness=True
                    )
                else:
                    loss = F.cross_entropy(logits.view(-1, Config.vocab_size), Y_batch.view(-1))
                    raw_loss = loss
        
            loss.backward() 
            step_loss = loss.item() 
            raw_lossA = raw_loss.item()

            raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.max_norm)

            if step % 200 == 0:
                loss_str = f"{step_loss:.4f}" if not math.isnan(step_loss) else "NaN"
                raw_str = f"{raw_lossA:.4f}" if not math.isnan(raw_lossA) else "NaN"
                logger.info(f"    Step {step:4d}/1000 | Raw Loss: {raw_str} | Comb Loss No: {loss_str}")
            
            if step % 500 == 0:
                base_mod = model._orig_mod if hasattr(model, '_orig_mod') else model
                layer_0_norm = get_layer_grad_norm(base_mod.layers[0])   
                last_layer_norm = get_layer_grad_norm(base_mod.layers[-1])
            
                logger.info(f"    Step {step:4d}/1000 | Raw Loss: {raw_str} | Comb Loss No: {loss_str} | Pre-Clip Grad Norm: {raw_grad_norm:.2f} | Layer 0 Grad: {layer_0_norm:6.4f} | Last Layer Grad: {last_layer_norm:6.4f}")
            
                global_step = (epoch - 1) * steps_per_epoch + step
                norms = [get_layer_grad_norm(base_mod.layers[i]) for i in range(Config.num_layers)]
                csv_buffer.append([global_step] + [f"{n:.4f}" for n in norms])
                
                profile_str = " | ".join([f"L{i}: {norms[i]:.4f}" for i in range(0, Config.num_layers, 3)])
                logger.info(f"    Step {global_step:4d} | Decile Profile -> {profile_str}")
            
            optimizer.step()
            scheduler.step()
            epoch_loss += step_loss
            epoch_raw += raw_lossA
            
        with open(csv_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_buffer)
            
        epoch_duration = time.time() - epoch_start_time
        steps_per_sec = steps_per_epoch / epoch_duration
        avg_train_loss = epoch_loss / steps_per_epoch
        avg_raw_loss = epoch_raw / steps_per_epoch
        
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            comb_loss = 0.0
            anisotropy_tensors = []  

            for _ in range(50): 
                X_val, Y_val = get_batch(val_data)
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    val_logits, final_hidden = model(X_val)

                    if Config.dpaloss:
                        base_model_ref = model._orig_mod if hasattr(model, '_orig_mod') else model
                        raw_val_loss, comb_loss_no = fused_nsm_loss(
                            val_logits, Y_val, epoch=epoch, 
                            poly_a=base_model_ref.loss_poly_a, 
                            poly_b=base_model_ref.loss_poly_b, 
                            poly_z_exp=base_model_ref.loss_poly_z_exp,
                            loss_bias=base_model_ref.loss_bias,
                            use_sharpness=True 
                        )
                        batch_val_loss = raw_val_loss
                    else:
                        batch_val_loss = F.cross_entropy(val_logits.view(-1, Config.vocab_size), Y_val.view(-1))
                        comb_loss_no = batch_val_loss

                val_loss += batch_val_loss.item()
                comb_loss += comb_loss_no.item()

                flat_hidden = final_hidden.view(-1, final_hidden.size(-1))
                norm_flat = F.normalize(flat_hidden, p=2, dim=-1)
                shifted_flat = torch.roll(norm_flat, shifts=1, dims=0)
                batch_anisotropy = (norm_flat * shifted_flat).sum(dim=-1).mean()
                anisotropy_tensors.append(batch_anisotropy)

            val_loss /= 50
            comb_loss /= 50
            if val_loss < val_low:
                val_low = val_loss

            mean_anisotropy = torch.stack(anisotropy_tensors).mean().item()
            
        val_loss_history[seed].append(val_loss)
        logger.info(f"Epoch {epoch:2d} | Train Loss Comb. No: {avg_train_loss:.4f} | Raw Train Loss: {avg_raw_loss:.4f} | Val Loss: {val_loss:.4f} | Penalty + Val Loss: {comb_loss:.4f} |Anisotropy: {mean_anisotropy:.3f} | Low Val Loss: {val_low:.4f}| Speed: {steps_per_sec:.1f} it/s")

        epoch_sample = generate_sample_text(model, enc)
        logger.info(f"    -> Sample: {epoch_sample}\n")

        if Config.attn_mode == 'dpa':
            logger.info(f"\n{'='*60}")
            logger.info(f" EPOCH {epoch} DPA POLYNOMIAL PROFILE")
            logger.info(f"{'='*60}")
    
            base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    
            for i, layer in enumerate(base_model.layers):
                attn = layer.attn
                a = F.softplus(attn.poly_a).item()
                b = F.softplus(attn.poly_b).item()
                z = 2.0 + F.softplus(attn.poly_z_exp).item()
        
                logger.info(f"Layer {i:2d} | a: {a:.4f} | b: {b:.4f} | z-exponent: {z:.4f}")
        
            logger.info(f"Loss Polynomial -> a: {F.softplus(base_model.loss_poly_a).item():.4f} | b: {F.softplus(base_model.loss_poly_b).item():.4f} | z-exponent: {2.0 + F.softplus(base_model.loss_poly_z_exp).item():.4f}")

    generated_samples[seed] = generate_sample_text(model, enc, max_new_tokens=100)

    if Config.attn_mode == 'dpa':
        logger.info(f"\n{'='*60}")
        logger.info(f" FINAL DPA POLYNOMIAL PROFILE")
        logger.info(f"{'='*60}")
    
        base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    
        for i, layer in enumerate(base_model.layers):
            attn = layer.attn
            a = F.softplus(attn.poly_a).item()
            b = F.softplus(attn.poly_b).item()
            z = 2.0 + F.softplus(attn.poly_z_exp).item()
        
            logger.info(f"Layer {i:2d} | a: {a:.4f} | b: {b:.4f} | z-exponent: {z:.4f}")
        
        logger.info(f"Loss Polynomial -> a: {F.softplus(base_model.loss_poly_a).item():.4f} | b: {F.softplus(base_model.loss_poly_b).item():.4f} | z-exponent: {2.0 + F.softplus(base_model.loss_poly_z_exp).item():.4f}")

    # Checkpoint Saving
    checkpoint_path = os.path.join(log_dir, f"model_{model_tag}_seed_{seed}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"[SYSTEM] Model saved to {checkpoint_path}")

# Final Report
logger.info(f"\n{'='*60}")
logger.info(f" {model_tag.replace('_', ' ').upper()} FINAL REPORT (AT EPOCH {Config.epochs})")
logger.info(f"{'='*60}")
final_losses = [losses[-1] for losses in val_loss_history.values()]
final_lossesmin = [min(losses) for losses in val_loss_history.values()]
mean_loss = np.mean(final_losses)
std_loss = np.std(final_losses)
mean_lossmin = np.mean(final_lossesmin)
std_lossmin = np.std(final_lossesmin)

logger.info(f"Final Validation Loss (Epoch {Config.epochs}) (Mean): {mean_loss:.4f} ± {std_loss:.4f}\n")
logger.info(f"BEST Validation Loss (All Epochs) (Mean): {mean_lossmin:.4f} ± {std_lossmin:.4f}\n")

for seed in seed_list:
    logger.info(f"--- Seed {seed} Sample ---")
    logger.info(generated_samples[seed])
    logger.info("-" * 30)