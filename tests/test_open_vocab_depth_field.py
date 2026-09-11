import numpy as np
import torch

from optollama.data.open_layer import MaterialCatalog, OpenVocabularyDepthFieldCollator
from optollama.model.open_vocab_depth_field import (
    OpenVocabularyDepthFieldConfig,
    OpenVocabularyDepthFieldDiffusion,
)


def tiny_model() -> OpenVocabularyDepthFieldDiffusion:
    """Return a CPU-sized open-vocabulary depth model."""
    return OpenVocabularyDepthFieldDiffusion(
        OpenVocabularyDepthFieldConfig(
            spectrum_shape=(3, 8),
            depth_bins=12,
            max_candidates=3,
            d_model=16,
            n_blocks=2,
            n_heads=2,
            kernel_size=3,
            hybrid_dilations=(1, 2),
            spectrum_patch_size=2,
            spectrum_patch_stride=2,
            spectrum_encoder_blocks=1,
            spectrum_encoder_heads=2,
        )
    )


def model_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Return a synthetic candidate-local training batch."""
    wavelengths = torch.linspace(300.0, 700.0, 8).expand(batch_size, -1)
    candidate_nk = torch.rand(batch_size, 3, 8, 2)
    candidate_mask = torch.tensor([[True, True, False]]).expand(batch_size, -1).clone()
    clean = torch.tensor([[0] * 4 + [1] * 3 + [3] * 5]).expand(batch_size, -1).clone()
    return {
        "wavelengths_nm": wavelengths,
        "target_spectrum": torch.rand(batch_size, 8, 3),
        "candidate_nk": candidate_nk,
        "candidate_mask": candidate_mask,
        "clean_fields": clean,
        "incidence_angle_deg": torch.zeros(batch_size),
        "polarization_id": torch.zeros(batch_size, dtype=torch.long),
    }


def test_collator_rasterizes_candidate_local_depth_field() -> None:
    """Map parsed layers to local candidate ids and a stable void class."""
    wavelengths = np.array([300.0, 500.0, 700.0])
    catalog = MaterialCatalog(
        names=("A", "B"),
        wavelengths_nm=(wavelengths, wavelengths),
        n_values=(np.ones(3), np.full(3, 2.0)),
        k_values=(np.zeros(3), np.zeros(3)),
    )
    collator = OpenVocabularyDepthFieldCollator(
        wavelengths_nm=torch.tensor(wavelengths),
        catalog=catalog,
        idx_to_token={0: "<PAD>", 1: "<MSK>", 2: "<EOS>", 3: "A_10", 4: "B_15"},
        eos_idx=2,
        pad_idx=0,
        msk_idx=1,
        channels=("R", "A", "T"),
        max_layers=4,
        max_candidates=2,
        min_query_points=3,
        max_query_points=3,
        query_sampling="full",
        randomize_candidates=False,
        random_distractors=False,
        dz_nm=5.0,
        max_total_nm=40.0,
        incidence_angle_deg=0.0,
        polarization="s",
    )
    spectrum = torch.rand(3, 3)
    batch = collator([(spectrum, torch.tensor([3, 4, 2]), 5)])
    assert batch["clean_fields"].tolist() == [[0, 0, 1, 1, 1, 2, 2, 2]]
    assert batch["incidence_angle_deg"].tolist() == [0.0]
    assert batch["polarization_id"].tolist() == [0]


def test_continuous_time_loss_and_sampling_are_finite() -> None:
    """Backpropagate at real-valued times and sample with an arbitrary grid."""
    torch.manual_seed(4)
    model = tiny_model()
    batch = model_batch()
    output = model.training_loss(batch, timesteps=torch.tensor([0.25, 0.75]))
    assert output["timesteps"].dtype == torch.float32
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.pointer_query.weight.grad is not None
    with torch.no_grad():
        samples = model.sample(
            spectra=batch["target_spectrum"].transpose(1, 2),
            wavelengths_nm=batch["wavelengths_nm"],
            candidate_nk=batch["candidate_nk"],
            candidate_mask=batch["candidate_mask"],
            incidence_angle_deg=batch["incidence_angle_deg"],
            polarization_id=batch["polarization_id"],
            steps=3,
            deterministic=True,
        )
    assert samples.shape == (2, 12)
    assert torch.all(samples != model.mask_id)
    assert torch.all((samples == 0) | (samples == 1) | (samples == model.void_id))


def test_candidate_permutation_preserves_physical_logits() -> None:
    """Make candidate order change labels without changing material semantics."""
    torch.manual_seed(5)
    model = tiny_model().eval()
    batch = model_batch(batch_size=1)
    fields = batch["clean_fields"].clone()
    time = torch.tensor([0.4])
    kwargs = {
        "wavelengths_nm": batch["wavelengths_nm"],
        "candidate_mask": batch["candidate_mask"],
        "incidence_angle_deg": batch["incidence_angle_deg"],
        "polarization_id": batch["polarization_id"],
    }
    with torch.no_grad():
        base = model(batch["target_spectrum"].transpose(1, 2), fields, time, candidate_nk=batch["candidate_nk"], **kwargs)
        permutation = torch.tensor([1, 0, 2])
        inverse = torch.tensor([1, 0, 2])
        permuted_fields = torch.where(fields < 3, inverse[fields.clamp_max(2)], fields)
        permuted = model(
            batch["target_spectrum"].transpose(1, 2),
            permuted_fields,
            time,
            candidate_nk=batch["candidate_nk"][:, permutation],
            candidate_mask=batch["candidate_mask"][:, permutation],
            wavelengths_nm=batch["wavelengths_nm"],
            incidence_angle_deg=batch["incidence_angle_deg"],
            polarization_id=batch["polarization_id"],
        )
    torch.testing.assert_close(base[..., :3], permuted[..., permutation])
    torch.testing.assert_close(base[..., 3], permuted[..., 3])


def test_normal_incidence_condition_is_polarization_invariant() -> None:
    """Keep s and p conditioning identical where they are physically degenerate."""
    features_s = OpenVocabularyDepthFieldDiffusion._optical_features(torch.tensor([0.0]), torch.tensor([0]))
    features_p = OpenVocabularyDepthFieldDiffusion._optical_features(torch.tensor([0.0]), torch.tensor([1]))
    torch.testing.assert_close(features_s, torch.zeros_like(features_s))
    torch.testing.assert_close(features_s, features_p)
