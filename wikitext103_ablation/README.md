Commercial and Non-Commercial Research & Development License
Copyright (c), 2026, Anthony Luke Simon. All rights reserved.

You are free to use the software to engage in all forms of Research and Development, commercial and non-commercial, including for academic and non-profit purposes and internal commercial purposes ("Purposes"). Purposes does not include commercial exploitation. Licensor hereby grants to the Licensee a worldwide, non-exclusive, non-transferable, royalty-free, revocable license to reproduce, modify, create derivative works from, and use the Software for the Purposes.  No other purposes are permitted.  To the extent patent rights exist or are obtained, no patent rights are granted beyond the Purposes.
Disclaimer of Warranty THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE LICENSOR BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
This License and the rights granted hereunder will terminate automatically if the Licensee fails to comply with any terms herein.


# Reproducibility Artifact: DPM Weight Decay Ablation Study

This repository contains the training scripts to reproduce the weight decay ablation studies presented in the paper. The codebase is configured to train a 24.7M parameter Softmax-free transformer (DPM) on subword BPE-tokenized WikiText-103 to evaluate the empirical impact of asymmetric regularization.

## 1. Environment Setup

**Prerequisites:**
- Python 3.8+
- PyTorch 2.0+ (CUDA recommended for torch.compile and bfloat16 AMP support)
- tiktoken (for GPT-2 BPE encoding)
- numpy, tqdm

Install command: pip install torch datasets tiktoken tqdm numpy

## 2. Dataset Preparation
**Dataset Requirements:**
Ensure the compiled dataset binaries (wiki_train.bin and wiki_val.bin) are located in the same root directory as the execution script.

An automated preparation script is included. This script downloads WikiText-103, tokenizes it using the GPT-2 vocabulary, and outputs the memory-mapped binary files (wiki_train.bin and wiki_val.bin) locally.

Run the data script prior to training:

	python prepare_data.py

## 3. Configuration & Execution Commands

The training script uses standard command-line flags to manage configurations, replacing manual code edits. Launch the script via your terminal using the appropriate command below:

### Configuration 1: Champion Model (Split-Delay Weight Decay)
This configuration safely delays weight decay on the core linear projections (by 75% of the total epochs) while immediately applying weight decay to the DPM polynomial coefficients. This allows the network to structurally anchor the activation curve without prematurely constraining feature exploration.

    python train.py --run_split_decay

### Configuration 2: Ablation Control (Simultaneous Decay)
This ablation forces all parameters (both linear weights and DPM polynomial coefficients) to undergo simultaneous weight decay starting at epoch 0. This demonstrates the destructive co-adaptation that occurs when both features and activation geometries are rigidly constrained at initialization.

    python train.py

## 4. Baseline Execution


The training script uses standard command-line flags to manage baseline selections, replacing manual code edits. Launch the script via your terminal using the appropriate flag below:

### Baseline 1: Hybrid Peri-LN with DPM Loss
This configuration isolates the Decoupled Polynomial Loss (DPL) mechanism by replacing Softmax at the vocabulary projection while maintaining standard Softmax in the attention mechanism.

    python train_baseline.py --baseline_mode hybrid_peri_ln

### Baseline 2: Sparse Prior Art (ReLU-Squared Attention)
This configuration tests the static ReLU-squared attention operator (with standard Pre-Norm) while retaining traditional cross-entropy (Softmax) for the loss projection.

    python train_baseline.py --baseline_mode relu_squared

### Baseline 3: Standard Sandwich Control (Standard Softmax)
This is the industry-standard control utilizing Softmax for both self-attention and cross-entropy loss, stabilized via Sandwich Normalization (Pre-Norm and Post-Norm).

    python train_baseline.py --baseline_mode standard


## 5. Outputs & Logging

Every execution automatically generates a timestamped run directory under ./logs/ containing:

1. Run Log (.txt): Step-by-step terminal metrics, dynamic loss tracking, and epoch-level text generation samples.

2. Gradient Profile (.csv): Layer-by-layer gradient norms captured periodically to monitor regularization dynamics and geometric compression.

3. Model Checkpoints (.pt): Final seed weights saved securely for further evaluation or downstream inference.
