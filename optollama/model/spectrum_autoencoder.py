from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from optollama.data.spectra import ensure_3w


class SpectrumAutoencoder(torch.nn.Module):
    """
    Compact autoencoder for RAT spectra.

    The model maps spectra of shape ``[B, 3, W]`` into a dense latent vector and
    decodes back to a valid RAT spectrum. The decoder uses a channel-wise
    softmax so ``R + A + T = 1`` at every wavelength.
    """

    def __init__(
        self,
        *,
        width: int,
        latent_dim: int = 128,
        hidden_dim: int = 1024,
        n_hidden: int = 2,
        dropout: float = 0.0,
        latent_bound: float = 3.0,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1, got {n_hidden}")

        self.width = int(width)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_hidden = int(n_hidden)
        self.dropout_p = float(dropout)
        self.latent_bound = float(latent_bound)
        input_dim = 3 * self.width

        self.encoder = self._make_mlp(input_dim, self.latent_dim, self.hidden_dim, self.n_hidden, self.dropout_p)
        self.decoder = self._make_mlp(self.latent_dim, input_dim, self.hidden_dim, self.n_hidden, self.dropout_p)

    @staticmethod
    def _make_mlp(input_dim: int, output_dim: int, hidden_dim: int, n_hidden: int, dropout: float) -> torch.nn.Sequential:
        layers: list[torch.nn.Module] = []
        dim = input_dim
        for _ in range(n_hidden):
            layers.extend(
                [
                    torch.nn.Linear(dim, hidden_dim),
                    torch.nn.SiLU(),
                    torch.nn.LayerNorm(hidden_dim),
                ]
            )
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(torch.nn.Linear(dim, output_dim))
        return torch.nn.Sequential(*layers)

    def arch_config(self) -> dict[str, Any]:
        """Return the constructor arguments needed to rebuild this model."""
        return {
            "width": self.width,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "n_hidden": self.n_hidden,
            "dropout": self.dropout_p,
            "latent_bound": self.latent_bound,
        }

    def encode(self, spectra: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, 3, W]`` spectra into latent vectors."""
        spectra, _ = ensure_3w(spectra)
        if spectra.dim() == 2:
            spectra = spectra.unsqueeze(0)
        if spectra.size(-1) != self.width:
            raise ValueError(f"Spectrum width mismatch: model W={self.width}, input W={spectra.size(-1)}")
        z = self.encoder(spectra.reshape(spectra.size(0), -1))
        if self.latent_bound > 0:
            z = self.latent_bound * torch.tanh(z / self.latent_bound)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into normalized ``[B, 3, W]`` spectra."""
        raw = self.decoder(z).reshape(z.size(0), 3, self.width)
        return torch.softmax(raw, dim=1)

    def forward(self, spectra: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, latent)`` for input spectra."""
        z = self.encode(spectra)
        return self.decode(z), z


def load_spectrum_autoencoder(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[SpectrumAutoencoder, dict[str, Any]]:
    """Load a saved :class:`SpectrumAutoencoder` checkpoint."""
    path = Path(checkpoint_path)
    blob = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(blob, dict) or "model_state" not in blob:
        raise ValueError(f"Invalid spectrum autoencoder checkpoint: {path}")

    arch = blob.get("arch")
    if not isinstance(arch, dict):
        raise ValueError(f"Checkpoint {path} does not contain an 'arch' dictionary.")

    model = SpectrumAutoencoder(**arch).to(device)
    model.load_state_dict(blob["model_state"], strict=True)
    model.eval()
    return model, blob


@lru_cache(maxsize=4)
def _cached_autoencoder(checkpoint_path: str, device_str: str) -> tuple[SpectrumAutoencoder, dict[str, Any]]:
    return load_spectrum_autoencoder(checkpoint_path, device=device_str)


@torch.no_grad()
def project_spectrum_with_autoencoder(
    spectrum: torch.Tensor,
    checkpoint_path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Encode/decode a target spectrum through a trained spectrum autoencoder.

    Returns a decoded spectrum on the original input device plus metadata with
    latent shape and reconstruction MAE against the provided input.
    """
    input_device = spectrum.device
    spectrum_3w, _ = ensure_3w(spectrum.to(torch.float32))
    if spectrum_3w.dim() == 2:
        spectrum_3w = spectrum_3w.unsqueeze(0)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = str(torch.device(device))
    model, blob = _cached_autoencoder(str(Path(checkpoint_path).resolve()), device_str)

    x = spectrum_3w.to(device_str)
    recon, z = model(x)
    mae = torch.mean(torch.abs(recon - x), dim=(1, 2))
    projected = recon.to(input_device)
    info = {
        "checkpoint": str(checkpoint_path),
        "mae_to_input": float(mae.mean().item()),
        "latent_dim": int(z.size(-1)),
        "width": int(projected.size(-1)),
        "epoch": blob.get("epoch"),
        "val_mae": float(blob["val_mae"]) if "val_mae" in blob else None,
    }
    return projected.squeeze(0), info
