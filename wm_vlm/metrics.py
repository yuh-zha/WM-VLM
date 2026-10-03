"""Distance metrics shared by WM-VLM training diagnostics and evaluation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


LATENT_METRIC_NAMES = (
    "mse",
    "rmse",
    "mae",
    "cosine",
    "pearson",
    "relative_l2",
    "normalized_mse",
    "token_cosine_mean",
    "token_cosine_std",
    "norm_ratio",
)


def latent_distance_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Return scale-sensitive and scale-free metrics for [tokens, hidden] latents."""
    if predicted.shape != target.shape:
        raise ValueError(
            f"Latent shapes differ: predicted={tuple(predicted.shape)}, "
            f"target={tuple(target.shape)}"
        )
    if predicted.ndim != 2:
        raise ValueError(
            f"Expected [tokens, hidden] latents, got {tuple(predicted.shape)}"
        )
    predicted = predicted.float()
    target = target.float()
    difference = predicted - target
    mse = difference.square().mean()
    target_energy = target.square().mean().clamp_min(1e-12)
    predicted_flat = predicted.flatten()
    target_flat = target.flatten()
    difference_norm = torch.linalg.vector_norm(difference)
    target_norm = torch.linalg.vector_norm(target).clamp_min(1e-12)
    predicted_norm = torch.linalg.vector_norm(predicted).clamp_min(1e-12)
    centered_predicted = predicted_flat - predicted_flat.mean()
    centered_target = target_flat - target_flat.mean()
    token_cosine = F.cosine_similarity(predicted, target, dim=-1, eps=1e-8)
    values = {
        "mse": mse,
        "rmse": mse.sqrt(),
        "mae": difference.abs().mean(),
        "cosine": F.cosine_similarity(
            predicted_flat.unsqueeze(0),
            target_flat.unsqueeze(0),
            dim=-1,
            eps=1e-8,
        )[0],
        "pearson": F.cosine_similarity(
            centered_predicted.unsqueeze(0),
            centered_target.unsqueeze(0),
            dim=-1,
            eps=1e-8,
        )[0],
        "relative_l2": difference_norm / target_norm,
        "normalized_mse": mse / target_energy,
        "token_cosine_mean": token_cosine.mean(),
        "token_cosine_std": token_cosine.std(unbiased=False),
        "norm_ratio": predicted_norm / target_norm,
    }
    result = {name: float(value.item()) for name, value in values.items()}
    if set(result) != set(LATENT_METRIC_NAMES):
        raise AssertionError("Latent metric schema changed unexpectedly")
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"Non-finite latent metric: {result}")
    return result
