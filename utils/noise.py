import torch
from torch.distributions import Gamma

# 噪声采样和 embedding 加噪

# copy from SnD
def sample_noise_Chi(d_shape, eta, device="cpu", dtype=None):
    n_dim = d_shape[-1]
    alpha = torch.ones(d_shape, device=device) * n_dim
    beta = torch.ones(d_shape, device=device) * eta
    m = Gamma(alpha, beta)
    l_lst = m.sample().to(device)
    v_lst = -2 * torch.rand(d_shape, device=device) + 1
    noise = l_lst * v_lst
    if dtype is not None:
        noise = noise.type(dtype)
    noise = noise.to(device)
    return noise


def sample_noise_Gaussian(d_shape, eta, delta=10e-5, device="cpu", dtype=None):
    sensitivity = 1.0 
    noise_stddev = (
        torch.sqrt(2 * torch.log(torch.tensor(1.25 / delta))) * sensitivity / eta
    )
    noise = torch.normal(mean=0.0, std=noise_stddev, size=d_shape, device=device)
    if dtype is not None:
        noise = noise.type(dtype)
    return noise


# copy from SnD
def get_noisy_embedding(
    embedding: torch.Tensor, eta: float, clip=False, noise_type="Chi", model_type="qwen2-1.5b"
):
    if eta <= 0:
        return embedding, torch.zeros((1,), device=embedding.device)

    if noise_type == "Chi":
        noise = sample_noise_Chi(embedding.shape, eta, device=embedding.device).type_as(
            embedding
        )
    elif noise_type == "Gaussian":
        noise = sample_noise_Gaussian(
            embedding.shape, eta, device=embedding.device
        ).type_as(embedding)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    noisy_embed = embedding + noise
    if not clip:
        return noisy_embed, noise

    if model_type == "qwen2-1.5b":
        max_norm = 0.974
    elif model_type == "t5-large":
        max_norm = 650
    elif model_type == "llama-3.2-1b":
        max_norm = 1.204
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        

    all_norms = torch.norm(noisy_embed, p=2, dim=-1)
    noisy_embed = noisy_embed * torch.clamp(max_norm / all_norms, max=1).unsqueeze(-1)
    noise = noisy_embed - embedding
    return noisy_embed, noise
