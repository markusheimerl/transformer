# transformer
A transformer implementation

Consider a transformer operating on batched sequences of shape (batch_size × seq_len × d_model). The architecture consists of alternating self-attention and multilayer perceptron layers with residual connections. Each transformer layer applies attention followed by a feed-forward network. The forward propagation for a single layer follows:

$$
\begin{align*}
Q &= XW_q \\
K &= XW_k \\
V &= XW_v \\
S &= \frac{QK^T}{\sqrt{d}} \\
A_{ij} &= \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})} \\
Z &= AV \\
\tilde{Z} &= ZW_o + X \\
H &= \tilde{Z}W_1 \\
S' &= H \odot \sigma(H) \\
Y &= S'W_2 + \tilde{Z}
\end{align*}
$$

The attention mechanism computes query, key, and value projections, applies scaled dot-product attention with softmax normalization, and projects the output. The first residual connection adds the input to the attention output. The MLP applies a linear transformation, swish activation, another linear transformation, and a second residual connection.

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```