# Variational Autoencoders for Generative Modeling
![GitHub last commit](https://img.shields.io/github/last-commit/elliothha/variational-autoencoders) ![GitHub repo size](https://img.shields.io/github/repo-size/elliothha/variational-autoencoders)

*[3/15/24 Update] Uploaded notebook for a VAE modeling the MNIST dataset*

This repo contains a PyTorch implementation of a standard Variational Autoencoder (VAE) purely intended for educational purposes.

Generated sample MNIST image [results](https://github.com/elliothha/variational-autoencoders/blob/main/README.md#after-30-training-epochs) after 30 training epochs

by **Elliot H Ha**. Duke University

[elliothha.tech](https://elliothha.tech/) | [elliot.ha@duke.edu](mailto:elliot.ha@duke.edu)

---

## Dependencies
- Jupyter Notebook
- PyTorch

## Project Structure
`models/` is the main folder containing the Jupyter Notebook file implementing the Variational Autoencoder model for the MNIST dataset. The raw dataset is stored in `models/data/MNIST/raw`.

## Hyperparameters & Architecture
```
lr = 1e-3
gamma = 0.1
step_size = 10
num_epochs = 30
batch_size = 100

input_dim = 784
hidden_dim = 400
latent_dim = 20
```

I use [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) as my optimizer with a learning rate, `lr = 1e-3`, default betas and epsilon, and 0 weight decay. I also use [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html) as my learning rate scheduler with a `step_size = 10` and `gamma = 0.1`.

The `input_dim` hyperparameter represents the full dimensionality of the flattened MNIST image, i.e., 28 * 28 = 784. 

The `hidden_dim` hyperparameter represents the dimensionality of the Encoder/Decoder's linear layers, with ReLU being used in-between linear layers as the activation function in both. 

From the Encoder, I split into two separate heads, one for the mean, mu, and one for the standard deviation, sigma. The output dimensionality of these latent variables is represented by `latent_dim`. No activation is used after this linear layer.

Given mu and sigma, I reparameterize the latent variable representation into a differentiable equation akin to using the change of variables formula.

Finally, I take the latent variable representation and pass it into the Decoder, which mirrors the Encoder + latent bottleneck architecture but in reverse. ReLU is used after each linear layer, and a sigmoid activation is used after the final one to map the logits back to [0, 1] as grayscale MNIST images are binary data.

Training is run for `num_epochs = 30` epochs with a `batch_size = 100`.

## VAE Generated Sample Results
### After 0 Training Epochs
Training loss: N/A, Validation loss: N/A
![VAE sampling results for 0 training epochs](/examples/samples_0.png)

### After 10 Training Epochs
Training loss: 9854.8673, Validation loss: 9807.5804
![VAE sampling results for 10 training epochs](/examples/samples_10.png)

### After 20 Training Epochs
Training loss: 9459.2032, Validation loss: 9486.7833
![VAE sampling results for 20 training epochs](/examples/samples_20.png)

### After 30 Training Epochs
Training loss: 9412.7533, Validation loss: 9456.9637
![VAE sampling results for 30 training epochs](/examples/large_samples_30.png)

### Training vs Validation Loss
![Training vs Validation loss](/examples/loss_plot.png)

---

## References
1. Auto-Encoding Variational Bayes, Kingma and Welling 2013 | [1312.6114](https://arxiv.org/abs/1312.6114)
2. Tutorial on Variational Autoencoders, Doersch 2016 | [1606.05908](https://arxiv.org/abs/1606.05908)