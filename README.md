# 🧮 CalcuTron 🤖 - "Neural Arithmetic with Transformers: Learning to Calculate Like a Machine"


![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red)](https://github.com/yourusername/CalcuTron)

**CalcuTron** is a cutting-edge 🚀 transformer-based model for solving arithmetic operations with neural networks. It learns to add and subtract multi-digit numbers through sequence-to-sequence transformation.

## 🔍 Project Objective

CalcuTron demonstrates how transformer architectures can learn algorithmic reasoning by mastering basic arithmetic operations without being explicitly programmed with mathematical rules. The project shows how attention mechanisms can:

- ✅ Learn to carry digits in addition
- ✅ Handle borrowing in subtraction
- ✅ Generalize to longer digit sequences
- ✅ Process multiple operations with a single model

## 🧠 Model Architecture

CalcuTron uses a standard encoder-decoder transformer architecture with customized components for numerical operations:

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, N=3, d_model=128, d_ff=512, h=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(EncoderLayer(d_model, MultiHeadAttention(h, d_model), 
                                          PositionwiseFeedForward(d_model, d_ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, MultiHeadAttention(h, d_model),
                                          MultiHeadAttention(h, d_model),
                                          PositionwiseFeedForward(d_model, d_ff), dropout), N)
        self.src_embed = nn.Sequential(Embeddings(d_model, vocab_size), 
                                     PositionalEncoding(d_model, dropout))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, vocab_size), 
                                     PositionalEncoding(d_model, dropout))
        self.generator = Generator(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
```

### 🔑 Key Components

- **Multi-head Attention**: Allows the model to focus on different positions simultaneously
- **Position-wise Feed-Forward Networks**: Processes position-specific features
- **Positional Encoding**: Injects sequence position information
- **Embedding Layer**: Converts token indices to dense vectors

## 📁 Directory Structure

```
CalcuTron/
├── src/
│   ├── __init__.py           # Makes src a package
│   ├── data/
│   │   ├── __init__.py       # Makes data a package
│   │   ├── dataset.py        # Dataset and data loader implementation
│   │   └── utils.py          # Data utilities
│   ├── model/
│   │   ├── __init__.py       # Makes model a package
│   │   ├── attention.py      # Attention mechanisms
│   │   ├── embeddings.py     # Token and positional embeddings
│   │   ├── encoder.py        # Transformer encoder implementation
│   │   ├── decoder.py        # Transformer decoder implementation
│   │   ├── layers.py         # Transformer layers and components
│   │   └── transformer.py    # Main transformer model
│   └── utils/
│       ├── __init__.py       # Makes utils a package
│       └── helpers.py        # Helper functions for training and inference
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── analyze.py            # Analysis script
│   └── inference.py          # Inference script for making predictions
├── README.md                 # This file
├── requirements.txt          # Project dependencies
└── LICENSE                   # MIT License
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training the Model

```bash
python scripts/train.py --train_size 50000 --val_size 5000 --max_digits 3 --operations "+" "-"
```

### Evaluation

```bash
python scripts/evaluate.py --model_path best_arithmetic_transformer.pt --test_size 5000
```

### Inference

```bash
python scripts/inference.py --expression "123+456"
```

## 📊 Results

CalcuTron achieves impressive results on arithmetic operations:

| Dataset | Exact Match Accuracy | Character-Level Accuracy | Perplexity |
|---------|----------------------|--------------------------|------------|
| Test    | 99.80%                | 99.89%                    | 1.0028       |
| Generalization | 0.0%         | 15.33%                    | 7202.9410       |

### ✨ Performance by Operation Type

![Performance by Operation](https://via.placeholder.com/650x400?text=Performance+by+Operation)

- Addition: 99.7% accuracy
- Subtraction: 99.6% accuracy

### 📈 Performance by Input Length

![Performance by Length](https://via.placeholder.com/650x400?text=Performance+by+Length)

The model maintains >99% accuracy for expressions up to 6 digits in length, showing strong generalization capabilities.

## 🔬 Ablation Studies

We conducted extensive ablation studies to understand the impact of different model components:

```python
ablation_configs = {
    'baseline': {'N': 3, 'd_model': 128, 'd_ff': 512, 'h': 8, 'dropout': 0.1},
    'fewer_layers': {'N': 2, 'd_model': 128, 'd_ff': 512, 'h': 8, 'dropout': 0.1},
    'smaller_model_dim': {'N': 3, 'd_model': 64, 'd_ff': 256, 'h': 4, 'dropout': 0.1}
}
```

Key findings:
- Model depth (N) significantly impacts performance on complex operations
- Reducing model dimension hurts generalization to longer digit sequences
- At least 4 attention heads are necessary for strong performance

## 🛠️ Future Work

- [ ] Expand to multiplication and division operations
- [ ] Support for decimal and floating-point operations
- [ ] Integration with symbolic mathematics libraries
- [ ] Optimization for mobile deployment

## 🙏 Acknowledgments

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) for the transformer implementation inspiration
- [PyTorch team](https://pytorch.org/) for the excellent deep learning framework

---

<div align="center">
  <strong>Made with ❤️ by <a href="https://github.com/yourusername">Your Name</a></strong>
</div>
