# transformer
A transformer implementation

Consider a transformer operating on batched sequences of shape (batch_size × seq_len × d_model). The architecture consists of alternating self-attention and multilayer perceptron layers with residual connections. Each transformer layer applies attention followed by a feed-forward network. The forward propagation follows:

$$
\begin{align*}
Q_1 &= XW_{q,1} \\
K_1 &= XW_{k,1} \\
V_1 &= XW_{v,1} \\
S_1 &= \frac{Q_1K_1^T}{\sqrt{d}} \\
A_{1,ij} &= \frac{\exp(S_{1,ij})}{\sum_k \exp(S_{1,ik})} \\
Z_1 &= A_1V_1 \\
\tilde{Z}_1 &= Z_1W_{o,1} + X \\
H_1 &= \tilde{Z}_1W_{1,1} \\
S_1' &= H_1 \odot \sigma(H_1) \\
X_1 &= S_1'W_{2,1} + \tilde{Z}_1 \\
Q_2 &= X_1W_{q,2} \\
K_2 &= X_1W_{k,2} \\
V_2 &= X_1W_{v,2} \\
S_2 &= \frac{Q_2K_2^T}{\sqrt{d}} \\
A_{2,ij} &= \frac{\exp(S_{2,ij})}{\sum_k \exp(S_{2,ik})} \\
Z_2 &= A_2V_2 \\
\tilde{Z}_2 &= Z_2W_{o,2} + X_1 \\
H_2 &= \tilde{Z}_2W_{1,2} \\
S_2' &= H_2 \odot \sigma(H_2) \\
Y &= S_2'W_{2,2} + \tilde{Z}_2
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
