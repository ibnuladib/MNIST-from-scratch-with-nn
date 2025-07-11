# Neural Network for MNIST Classification (From Scratch)

A 2-layer neural network implementation for MNIST digit classification with full mathematical formulation.

## Network Architecture
- **Input layer**: 784 neurons (28×28 flattened MNIST images)
- **Hidden layer**: 10 neurons with ReLU activation
- **Output layer**: 10 neurons with Softmax activation

## Forward Propagation

### Mathematical Formulation

#### Layer 1 (Hidden Layer)
$$
\begin{aligned}
\mathbf{Z}^{[1]} &= \mathbf{W}^{[1]}\mathbf{X} + \mathbf{b}^{[1]} \\
\mathbf{A}^{[1]} &= g_{\text{ReLU}}(\mathbf{Z}^{[1]}) = \max(0, \mathbf{Z}^{[1]})
\end{aligned}
$$

#### Layer 2 (Output Layer)
$$
\begin{aligned}
\mathbf{Z}^{[2]} &= \mathbf{W}^{[2]}\mathbf{A}^{[1]} + \mathbf{b}^{[2]} \\
\mathbf{A}^{[2]} &= g_{\text{Softmax}}(\mathbf{Z}^{[2]}) = \frac{e^{\mathbf{Z}^{[2]}}}{\sum e^{\mathbf{Z}^{[2]}}}
\end{aligned}
$$

### Dimensionality
| Variable       | Dimensions  | Description                  |
|----------------|-------------|------------------------------|
| $\mathbf{X}$   | $784 \times m$ | Input matrix (m samples)    |
| $\mathbf{W}^{[1]}$ | $10 \times 784$ | Hidden layer weights       |
| $\mathbf{b}^{[1]}$ | $10 \times 1$   | Hidden layer biases        |
| $\mathbf{Z}^{[1]}, \mathbf{A}^{[1]}$ | $10 \times m$ | Hidden layer activations |
| $\mathbf{W}^{[2]}$ | $10 \times 10$  | Output layer weights      |
| $\mathbf{b}^{[2]}$ | $10 \times 1$   | Output layer biases       |
| $\mathbf{Z}^{[2]}, \mathbf{A}^{[2]}$ | $10 \times m$ | Output probabilities     |

## Backward Propagation

### Gradient Calculations

#### Output Layer Gradients
$$
\begin{aligned}
d\mathbf{Z}^{[2]} &= \mathbf{A}^{[2]} - \mathbf{Y} \\
d\mathbf{W}^{[2]} &= \frac{1}{m} d\mathbf{Z}^{[2]} \mathbf{A}^{[1]T} \\
d\mathbf{b}^{[2]} &= \frac{1}{m} \sum_{\text{axis}=1} d\mathbf{Z}^{[2]}
\end{aligned}
$$

#### Hidden Layer Gradients
$$
\begin{aligned}
d\mathbf{Z}^{[1]} &= \mathbf{W}^{[2]T} d\mathbf{Z}^{[2]} \odot g'_{\text{ReLU}}(\mathbf{Z}^{[1]}) \\
d\mathbf{W}^{[1]} &= \frac{1}{m} d\mathbf{Z}^{[1]} \mathbf{X}^T \\
d\mathbf{b}^{[1]} &= \frac{1}{m} \sum_{\text{axis}=1} d\mathbf{Z}^{[1]}
\end{aligned}
$$

Where:
- $\odot$ denotes element-wise multiplication
- $g'_{\text{ReLU}}(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}$

### Gradient Dimensions
| Gradient       | Dimensions  | Description                  |
|----------------|-------------|------------------------------|
| $d\mathbf{Z}^{[2]}$ | $10 \times m$ | Output layer error         |
| $d\mathbf{W}^{[2]}$ | $10 \times 10$ | Weight gradients          |
| $d\mathbf{b}^{[2]}$ | $10 \times 1$  | Bias gradients            |
| $d\mathbf{Z}^{[1]}$ | $10 \times m$ | Hidden layer error        |
| $d\mathbf{W}^{[1]}$ | $10 \times 784$ | Weight gradients         |
| $d\mathbf{b}^{[1]}$ | $10 \times 1$  | Bias gradients           |

## Parameter Update

### Update Rules
$$
\begin{aligned}
\mathbf{W}^{[2]} &:= \mathbf{W}^{[2]} - \alpha d\mathbf{W}^{[2]} \\
\mathbf{b}^{[2]} &:= \mathbf{b}^{[2]} - \alpha d\mathbf{b}^{[2]} \\
\mathbf{W}^{[1]} &:= \mathbf{W}^{[1]} - \alpha d\mathbf{W}^{[1]} \\
\mathbf{b}^{[1]} &:= \mathbf{b}^{[1]} - \alpha d\mathbf{b}^{[1]}
\end{aligned}
$$

Where $\alpha$ is the learning rate.
