# 🧸 DPM-Transformer: Look Mom, No Softmax!

**A 25M parameter causal language model utilizing Decoupled Polynomial Mapping (DPM) and true mathematical sparsity.**

This repository contains the model architecture and training scripts for a 15-layer DPM language model transformer. This architecture removes standard exponential Softmax in both self-attention and loss computation, replacing it with DPM: a ReLU activation paired with a learned multi-term polynomial expansion.

> **Abstract:** Decoupled Polynomial Mapping (DPM) replaces standard exponential Softmax operations in self-attention and loss computation in a language model transformer. DPM implements a learnable multi-term polynomial, $a \cdot r^2 + b \cdot r^z$ (where $r$ represents the rectified attention scores). By optimizing the coefficients $a$, $b$, and $z$ during training, DPM enables layer-specific learned activation curves. *(Note: Benchmarks and pre-trained weights for the 25M-parameter DPM transformer are currently being re-evaluated following a causal masking fix.)*

---

## 📊 Empirical Proof: The Pareto Frontier
To provide an objective comparison, **Next-Token Accuracy** and **WikiText-103 Zero-Shot Generalization** are used as benchmarks to compare model performance.

| Model Config | Seed | Batch Size | OWT Val Loss | OWT Next-Token Acc | Wiki-103 Perplexity | Wiki-103 Zero-Shot Acc | HellaSwag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Softmax Baseline** | 105 | 16 | 4.04 | 30.88% | 179.38 | 23.47% | 25.52% |
| **DPM** | *TBD* | 16 | *In progress* | *In progress* | *In progress* | *In progress* | *In progress* |

---

## 🏗️ Architecture Specifications

- **Topology:** 15 Layers, 4 Attention Heads
- **Dimensionality:** 256 Embedding Dim, 1024 FFN Dim
- **Context Window:** 512 Tokens
- **Positional Embeddings:** RoPE (Rotary Position Embedding)
- **Normalization:** RMSNorm (Post-Norm Configuration + Q/K Normalization)
- **Attention Activation:** ReLU and learned polynomial expansion (DPM)
- **Parameter Count:** 25 Million
- **Training Data:** 1 Epoch on OpenWebText

---

## 📂 Repository Contents

- [`wikitext103_ablation/`](./wikitext103_ablation/): Complete PyTorch training scripts (`prepare_data.py`, `train.py`, `train_baseline.py`) and instructions to reproduce the 24.7M-parameter WikiText-103 weight-decay ablation and baseline comparisons.
## 📜 License

**Commercial and Non-Commercial Research & Development License**  
Copyright (c), 2026, Anthony Luke Simon. All rights reserved.

* #LICENSE
Commercial and Non-Commercial Research & Development License
Copyright (c), 2026, Anthony Luke Simon. All rights reserved.

You are free to use the software to engage in all forms of Research and Development, commercial and non-commercial, including for academic and non-profit purposes and internal commercial purposes ("Purposes"). Purposes does not include commercial exploitation. Licensor hereby grants to the Licensee a worldwide, non-exclusive, non-transferable, royalty-free, revocable license to reproduce, modify, create derivative works from, and use the Software for the Purposes.  No other purposes are permitted.  To the extent patent rights exist or are obtained, no patent rights are granted beyond the Purposes.
Disclaimer of Warranty THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE LICENSOR BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
This License and the rights granted hereunder will terminate automatically if the Licensee fails to comply with any terms herein.


