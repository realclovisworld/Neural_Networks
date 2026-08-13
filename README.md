# Neural Networks: From Foundations to Frontiers

> A comprehensive collection of implementations, visualizations, and educational resources for understanding Neural Networks — from the Perceptron to Transformers.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Neural Network Architectures](#neural-network-architectures)
  - [Multi-Layer Perceptron (MLP)](#1-multi-layer-perceptron-mlp)
  - [Convolutional Neural Network (CNN)](#2-convolutional-neural-network-cnn)
  - [Recurrent Neural Network (RNN / LSTM)](#3-recurrent-neural-network-rnn--lstm)
  - [Transformer](#4-transformer)
- [Key Concepts & Diagrams](#key-concepts--diagrams)
- [Textbooks & References](#textbooks--references)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository serves as a practical and theoretical guide to Neural Networks. It includes:

- **From-scratch implementations** of core architectures (NumPy & PyTorch)
- **Interactive visualizations** of forward/backward propagation, attention mechanisms, and gradient flow
- **Jupyter notebooks** with step-by-step explanations
- **References to seminal textbooks** for deep theoretical grounding

Whether you are a student beginning your journey in Deep Learning or a practitioner seeking to solidify your understanding of foundational concepts, this repository is designed for you.

---

## Repository Structure

```
neural-networks/
├── notebooks/
│   ├── 01_perceptron_and_mlp.ipynb
│   ├── 02_cnn_from_scratch.ipynb
│   ├── 03_rnn_and_lstm.ipynb
│   ├── 04_transformer_attention.ipynb
│   └── 05_training_dynamics.ipynb
├── src/
│   ├── layers/
│   ├── models/
│   └── utils/
├── diagrams/
│   ├── mlp_architecture.png
│   ├── cnn_feature_maps.png
│   ├── rnn_unrolled.png
│   └── transformer_block.png
├── data/
├── tests/
└── README.md
```

---

## Neural Network Architectures

### 1. Multi-Layer Perceptron (MLP)

The **Multi-Layer Perceptron** is the foundational feedforward neural network. It consists of an input layer, one or more hidden layers, and an output layer. Each neuron applies a weighted sum of its inputs followed by a non-linear activation function.

```
    Input Layer          Hidden Layer 1        Hidden Layer 2       Output Layer
    (Features)

       x₁ ──●──────╮
              ╲     │
       x₂ ──●──────┼──●──────╮
              ╲     │    ╲    │
       x₃ ──●──────┼──●──────┼──●──────╮
              ╲     │    ╲    │    ╲    │
       x₄ ──●──────╯      ╲   │      ╲  │
                              ●──────────┼──●──→  ŷ
       x₅ ──●──────╮      ╱   │      ╱  │
              ╲     │    ╱    │    ╱    │
       x₆ ──●──────┼──●──────╯    ╱     │
              ╲     │              ╱      │
       x₇ ──●──────╯         ●───────────╯

    [Weights W¹]        [Weights W²]         [Weights W³]
    [Bias b¹]           [Bias b²]            [Bias b³]
```

**Key Concepts:**
- **Forward Propagation:** $\mathbf{z}^{[l]} = \mathbf{W}^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$, followed by activation $\mathbf{a}^{[l]} = g(\mathbf{z}^{[l]})$
- **Backpropagation:** Chain rule applied to compute gradients $rac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$ and $rac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}}$
- **Universal Approximation Theorem:** A feedforward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of $\mathbb{R}^n$.

---

### 2. Convolutional Neural Network (CNN)

**Convolutional Neural Networks** are specialized for processing grid-like data such as images. They use convolutional layers that apply learnable filters to local regions of the input, capturing spatial hierarchies of features.

```
    Input Image (3×H×W)          Feature Maps          Feature Maps

    ┌─────────────┐
    │  R  G  B    │
    │  ┌───────┐  │    ┌─────┐    ┌─────┐    ┌─────┐
    │  │Kernel │  │───→│ ◼◼◼ │───→│ ◼◼◼ │───→│ ◼◼◼ │
    │  │ 3×3   │  │    │ ◼◼◼ │    │ ◼◼◼ │    │ ◼◼◼ │
    │  └───────┘  │    └─────┘    └─────┘    └─────┘
    │             │      │           │           │
    └─────────────┘      ▼           ▼           ▼
                      Conv1       ReLU+Pool    Conv2

    [Edge/Color    [Texture/      [Object       [Classification
     Detection]     Pattern]        Parts]         Score]
```

**Key Concepts:**
- **Convolution Operation:** $(I * K)(i,j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m,n)$
- **Parameter Sharing:** The same filter is applied across the entire input, drastically reducing the number of parameters.
- **Translation Equivariance:** If the input shifts, the feature map shifts correspondingly — a property crucial for image understanding.
- **Pooling:** Downsampling (MaxPool, AvgPool) reduces spatial dimensions and provides a degree of translation invariance.

---

### 3. Recurrent Neural Network (RNN / LSTM)

**Recurrent Neural Networks** process sequential data by maintaining a hidden state that captures information about previous inputs. Standard RNNs suffer from vanishing gradients, which **Long Short-Term Memory (LSTM)** networks address through a sophisticated gating mechanism.

```
    Standard RNN (Unrolled)

    x₁ ──→[  RNN  ]──→ h₁ ──→[  RNN  ]──→ h₂ ──→[  RNN  ]──→ h₃
              ↑                  ↑                  ↑
            h₀ (init)          h₁                 h₂

    ─────────────────────────────────────────────────────────────

    LSTM Cell Architecture

              ┌─────────────────────────────────────┐
              │           Forget Gate: fₜ            │
              │    fₜ = σ(W_f · [hₜ₋₁, xₜ] + b_f)   │
              │              ────→                   │
              │           Input Gate: iₜ             │
              │    iₜ = σ(W_i · [hₜ₋₁, xₜ] + b_i)   │
    xₜ ──→    │              ────→                   │    ──→ hₜ
              │         Candidate: C̃ₜ              │
              │    C̃ₜ = tanh(W_C · [hₜ₋₁, xₜ] + b_C)│
              │              ────→                   │
              │     Cell State Update: Cₜ            │
              │    Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ       │
              │              ────→                   │
              │         Output Gate: oₜ             │
              │    oₜ = σ(W_o · [hₜ₋₁, xₜ] + b_o)   │
              │    hₜ = oₜ ⊙ tanh(Cₜ)               │
              └─────────────────────────────────────┘
```

**Key Concepts:**
- **Hidden State:** $\mathbf{h}_t = 	anh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$
- **LSTM Gates:** Forget gate (what to discard), Input gate (what to store), Output gate (what to output)
- **Cell State:** The "conveyor belt" that runs through the entire chain with minimal interaction, allowing long-range dependencies
- **Bidirectional RNNs:** Process sequences in both directions to capture future context

---

### 4. Transformer

The **Transformer** architecture, introduced in *"Attention Is All You Need"* (Vaswani et al., 2017), revolutionized Deep Learning by replacing recurrence with **self-attention**, enabling parallelization and capturing long-range dependencies effectively.

```
    Transformer Encoder Block

    Input Embeddings + Positional Encoding
              │
              ▼
    ┌─────────────────────┐
    │   Multi-Head        │
    │   Self-Attention    │◄──── Q, K, V from same input
    │                     │      (Scaled Dot-Product Attention)
    │   Attention(Q,K,V)  │
    │   = softmax(QKᵀ/√dₖ)V
    └──────────┬──────────┘
               │
         Add & Norm (Residual + LayerNorm)
               │
               ▼
    ┌─────────────────────┐
    │   Feed-Forward      │
    │   Network (FFN)     │
    │   (Linear → ReLU    │
    │    → Linear)        │
    └──────────┬──────────┘
               │
         Add & Norm (Residual + LayerNorm)
               │
               ▼
           Output

    ─────────────────────────────────────────────────────────

    Multi-Head Attention (Parallel Attention Heads)

         Input
           │
    ┌──────┼──────┬──────┬──────┐
    │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼
   Head1  Head2  Head3  Head4  Head5
    │      │      │      │      │
    └──────┴──────┴──────┴──────┘
              │
         Concatenate
              │
         Linear Projection
              │
           Output
```

**Key Concepts:**
- **Self-Attention:** $	ext{Attention}(Q, K, V) = 	ext{softmax}\left(rac{QK^T}{\sqrt{d_k}}
ight)V$
- **Multi-Head Attention:** Projects Q, K, V into $h$ different subspaces, allowing the model to attend to information from different representation spaces simultaneously.
- **Positional Encoding:** Injects sequence order information since the architecture itself is permutation-invariant: $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
- **Layer Normalization:** Normalizes across the feature dimension, stabilizing training of very deep networks.

---

## Key Concepts & Diagrams

### The Forward-Backward Computation Graph

```
    Forward Pass                          Backward Pass
    ────────────                          ────────────

    x ──→ [Linear] ──→ z ──→ [ReLU] ──→ a ──→ [Loss] ──→ L
              ↑                                              │
              W                                              ▼
              │                                         ∂L/∂W
              │                                              │
              │         ∂L/∂a · ∂a/∂z · ∂z/∂W  ◄───────────┘
              │              (Chain Rule)
              │
              ▼
         Gradient Update
         W ← W - η·∂L/∂W
```

### Gradient Flow: Vanishing vs. Exploding Gradients

```
    Vanishing Gradients (Sigmoid/Tanh)          Exploding Gradients (Unbounded)
    ─────────────────────────────────           ───────────────────────────────

    ∂L/∂W₁ ≈ 0.0001                              ∂L/∂W₁ ≈ 10⁸
         │                                            │
    ∂L/∂W₂ ≈ 0.001                              ∂L/∂W₂ ≈ 10⁶
         │                                            │
    ∂L/∂W₃ ≈ 0.01                               ∂L/∂W₃ ≈ 10⁴
         │                                            │
    ∂L/∂W₄ ≈ 0.1                                ∂L/∂W₄ ≈ 10²
         │                                            │
    ∂L/∂W₅ ≈ 0.5                                ∂L/∂W₅ ≈ 10⁰

    [Solutions: ReLU, Residual Connections,    [Solutions: Gradient Clipping,
     BatchNorm, Xavier/He Initialization]         Weight Regularization]
```

### Activation Functions

```
    ReLU:        max(0, x)              Leaky ReLU:   max(αx, x)
    ┌──┐                              ┌──┐
    │ ╱                               │ ╱
    │╱                                │╱
    └──┬──────→                      └──┬──────→
       0                                0

    Sigmoid:     1/(1+e⁻ˣ)            Tanh:         (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)
    ┌─────┐                            ┌─────┐
    │    ╱│                            │    ╱│
    │   ╱ │                            │   ╱ │
    │  ╱  │                            │  ╱  │
    └──┴──┴──→                         └──┴──┴──→
       0                                  0
```

---

## Textbooks & References

### Foundational Textbooks

| # | Title | Authors | Year | Key Focus |
|---|-------|---------|------|-----------|
| 1 | **Deep Learning** | Ian Goodfellow, Yoshua Bengio, Aaron Courville | 2016 | The definitive comprehensive reference. Covers MLPs, CNNs, RNNs, optimization, regularization, and generative models. *[Available free at deeplearningbook.org](https://www.deeplearningbook.org/)* |
| 2 | **Neural Networks and Deep Learning** | Michael Nielsen | 2019 | An exceptionally clear, free online book. Ideal for beginners. Builds intuition for backpropagation and network design through interactive visualizations. |
| 3 | **Pattern Recognition and Machine Learning** | Christopher M. Bishop | 2006 | A rigorous treatment of probabilistic graphical models, Bayesian methods, and kernel methods. Essential for understanding the statistical foundations. |
| 4 | **The Elements of Statistical Learning** | Trevor Hastie, Robert Tibshirani, Jerome Friedman | 2009 | The classic reference for statistical learning theory. Covers neural networks alongside SVMs, trees, and regularization. *[Available free at authors' website](https://hastie.su.domains/ElemStatLearn/)* |
| 5 | **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** | Aurélien Géron | 2022 | Practical, code-first approach. Excellent for implementing CNNs, RNNs, and Transformers using modern frameworks. |
| 6 | **Dive into Deep Learning** | Aston Zhang, Zachary C. Lipton, Mu Li, Alex J. Smola | 2023 | A fully interactive book with PyTorch, TensorFlow, and MXNet code. Covers attention mechanisms and modern architectures in depth. *[Available free at d2l.ai](https://d2l.ai/)* |
| 7 | **Understanding Deep Learning** | Simon J.D. Prince | 2023 | A visually rich, modern textbook that explains deep learning concepts with exceptional clarity. Strong focus on Transformers and generative models. *[Available free at udlbook.com](https://udlbook.com/)* |
| 8 | **Mathematics for Machine Learning** | Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong | 2020 | The mathematical foundations — linear algebra, calculus, probability, and optimization — presented specifically for ML practitioners. *[Available free at mml-book.com](https://mml-book.com/)* |

### Seminal Papers

| Paper | Authors | Year | Contribution |
|-------|---------|------|-------------|
| **"Learning representations by back-propagating errors"** | Rumelhart, Hinton, Williams | 1986 | Popularized backpropagation for training multi-layer networks. |
| **"ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet)** | Krizhevsky, Sutskever, Hinton | 2012 | Sparked the modern deep learning revolution with CNNs. |
| **"Deep Residual Learning for Image Recognition" (ResNet)** | He et al. | 2016 | Introduced skip connections, enabling training of networks with 100+ layers. |
| **"Attention Is All You Need" (Transformer)** | Vaswani et al. | 2017 | The foundational paper for the Transformer architecture. |
| **"BERT: Pre-training of Deep Bidirectional Transformers"** | Devlin et al. | 2019 | Demonstrated the power of pre-training with bidirectional context. |
| **"A Simple Method for Commonsense Reasoning" (Chain-of-Thought)** | Wei et al. | 2022 | Pioneered prompting strategies that unlocked reasoning in large language models. |

### Online Courses & Lectures

- **[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)** — Stanford (Fei-Fei Li, Justin Johnson, Serena Yeung)
- **[CS224n: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)** — Stanford (Christopher Manning)
- **[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)** — Coursera (Andrew Ng)
- **[Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)** — Andrej Karpathy (YouTube)

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ (or TensorFlow 2.x)
- NumPy, Matplotlib, Jupyter

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-networks.git
cd neural-networks

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running a Notebook

```bash
jupyter notebook notebooks/01_perceptron_and_mlp.ipynb
```

### Quick Example: MLP with PyTorch

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Initialize model
model = MLP(input_dim=784, hidden_dim=256, output_dim=10)
print(model)
```

---

## Contributing

Contributions are welcome! Whether it's fixing a typo, adding a new architecture, improving visualizations, or expanding the references — every contribution helps make this a better resource for the community.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-addition`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-addition`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate documentation.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The textbooks and papers referenced herein remain under their respective authors' copyrights. Links provided are to official, freely available versions where applicable.

---

<div align="center">

**"The brain is a machine, and we can build one."** — *Geoffrey Hinton*

⭐ If you find this repository helpful, please consider giving it a star!

</div>
