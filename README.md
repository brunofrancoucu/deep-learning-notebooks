# Deep Learning Notebooks

The course introduces the fundamentals of neural networks and deep learnon. We will follow closely the book **"Hands-On Machine Learning with Scikit-Learn and PyTorch"** by Aurélien Géron.

### 1. Install VS Code / Cursor extensions

Install these two extensions:

- **Python** (`ms-python.python`) — Python language support
- **Jupyter** (`ms-toolsai.jupyter`) — notebook support, including kernel selection and cell execution

In VS Code/Cursor: open the Extensions panel (`Cmd+Shift+X`), search for each name, and click **Install**.

### 2. Install dependencies

This project uses [`uv`](https://docs.astral.sh/uv/) as the package manager.
If you don't have it yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the repo root, create a virtual environment and install all dependencies (including `ipykernel`, which lets VS Code/Cursor talk to the Python kernel):

```bash
uv sync
uv add ipykernel
```

This reads `pyproject.toml` and installs `torch`, `torchvision`, `matplotlib`, and `ipykernel` into `.venv/`.

## Content

### 1. Supervised learning

- **Optional** section at the end where you apply linear regression to FashionMNIST images and see what happens.

#### To be Learnt:

1. **Statistical intuitions with PyTorch** — sample from distributions, estimate expectations empirically, and observe how estimation error shrinks as you collect more data (the Law of Large Numbers).
2. **Linear regression from scratch** — implement MSE loss and gradient descent by hand on synthetic data, then visualize the loss landscape and training trajectory.
3. **Linear regression on a real dataset** — apply the same ideas to the California Housing dataset using `torch.nn` and `torch.optim`.

> Reference [lab_1.ipynb](/lab_1.ipynb)
---

### 2. Shallow neural networks

- **Shallow neural networks**. By adding a single hidden layer with a nonlinear activation function.
- The **Universal Approximation Theorem** guarantees that, with enough hidden units, a shallow network can get arbitrarily close to any continuous function.

#### To be Learnt:

1. Visualize how **ReLU activations** create piecewise linear functions — the \joints\ from Lecture 3.
2. Implement and compare **activation functions** (ReLU, Leaky ReLU, Sigmoid, Tanh).
3. Build a **shallow neural network** using `nn.Module`.
4. Train it on **FashionMNIST** and compare against the linear baseline from Notebook 1.

> Reference [lab_2.ipynb](/lab_2.ipynb)
---

### 3. Deep neural networks

- **Deep neural networks** — networks with multiple hidden layers.
- Which can produce up to $(D+1)^K$ linear regions, compared to only $D+1$ for a shallow network. This exponential increase in expressiveness is the key advantage of depth.

#### To be Learnt:

1. **Compose** two shallow networks and visualize the \"folding\" effect on the input space.
2. **Inspect** network architectures using `torchinfo` — counting parameters and understanding layer shapes.
3. **Design** deep architectures under a fixed parameter budget and compare their performance.
4. **Visualize** learned representations using PCA to understand what depth does internally.
5. **Evaluate** whether depth helps equally on structured (images) vs. tabular data.

> Reference [lab_3.ipynb](/lab_3.ipynb)
---

### 4. Loss functions and optimizers

- **Loss functions** arise from the maximum likelihood recipe: choose a probability distribution over outputs, and minimize the negative log-likelihood.
- **Optimizers** vary widely in how they navigate the loss landscape, and that **initialization** can make or break training in deep networks.

#### To be Learnt:
    
1. **Derive** loss functions from the maximum likelihood recipe and implement them from scratch
2. **Inspect** gradients computed by backpropagation and verify them manually
3. **Compare** SGD, SGD with momentum, and Adam on the same architecture
4. **Diagnose** vanishing/exploding gradients and fix them with He initialization
5. **Experiment** with learning rates, batch sizes, and learning rate schedules

> Reference [lab_4.ipynb](/lab_4.ipynb)
---

### 5. Measuring performance and regularization

- **UCI Covertype**, predicting forest cover type from cartographic features — where regularization pays off

#### To be Learnt:

1. Build a proper train / validation / test split in PyTorch, including stratified splits and per-class diagnostics.
2. Tune hyperparameters with `sklearn.model_selection.ParameterGrid` and with **Optuna**.
3. Compare regularization techniques — weight decay (AdamW), dropout, label smoothing, early stopping — under a fair protocol.
4. Use learning-rate schedulers (`StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`) and pick the right one for the situation.
5. *(Optional)* Observe **double descent** — test error as a function of model width, past the interpolation threshold.

> Reference [lab_5.ipynb](/lab_5.ipynb)
---

### 6. Convolutional and residual networks
#### To be Learnt:

1. **Derive** loss functions from the maximum likelihood recipe and implement them from scratch.
2. **Inspect** gradients computed by backpropagation and verify them manually.
3. **Compare** SGD, SGD with momentum, and Adam on the same architecture.
4. **Diagnose** vanishing/exploding gradients and fix them with He initialization.
5. **Experiment** with learning rates, batch sizes, and learning rate schedules.

> Reference [lab_6.ipynb](/lab_6.ipynb)
---

### 7. Transformers and attention mechanisms

- Build **decoder-only transformer** (a "mini-GPT") **from scratch** — no `nn.MultiheadAttention`, no `nn.Transformer`* 
- Train character by character on **Tiny Shakespeare** document.

#### To be Learnt:

1. Implement **scaled dot-product attention** from primitive `torch` ops.
2. Build **causal multi-head self-attention** from scratch (Q/K/V projections, head reshape, causal mask).
3. Assemble a **transformer block** (pre-LN) and stack them into a small GPT.
4. Train the model with a plain PyTorch loop and **generate text** autoregressively (temperature + top-k).
5. **Visualize per-head attention** matrices and interpret the patterns each head learns.
6. *(Optional)* Re-express attention with `torch.einsum`, and refactor the training loop into a **PyTorch Lightning** module.

> Reference [lab_7.ipynb](/lab_7.ipynb)

---

### 8. Generative Models

- **Variational Autoencoders (VAEs)** — encode an image to a low-dimensional Gaussian latent, decode back, and train with the ELBO (reconstruction + KL).
- **Denoising Diffusion Probabilistic Models (DDPM)** — corrupt an image with Gaussian noise over 
 steps, then train a U-Net to undo the corruption one step at a time.

#### To be Learnt:

1. Implement a **Gaussian VAE** — encoder, reparameterization trick, Bernoulli decoder, and the ELBO loss (BCE reconstruction + closed-form KL).
2. **Sample** from the VAE prior and interpolate between two real images in latent space.
3. Implement DDPM's **forward noising process** using the closed-form marginal.
3. Build a small **U-Net** with sinusoidal time embedding and train it with the noise-prediction MSE objective.
5. Implement the **DDPM reverse sampling loop** and visualize the denoising trajectory.
6. Compare VAE and DDPM samples on the same dataset — connecting back to lecture §6's trade-off table.
7. (Optional) Run a pretrained DDPM from HuggingFace `diffusers`, swap the scheduler from DDPM to DDIM, and see the ~20× speedup at similar quality.

> Reference [lab_8.ipynb](/lab_8.ipynb)