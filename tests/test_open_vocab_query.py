from __future__ import annotations

import numpy as np
import torch

from optollama.data.open_layer import (
    MaterialCatalog,
    OpenVocabularyDepthFieldCollator,
    coordinate_query_wavelengths,
    interpolate_query_spectra,
)
from optollama.model.open_vocab_depth_field import OpenVocabularyDepthFieldConfig, OpenVocabularyDepthFieldDiffusion
from scripts.inference_open_vocab_depth_field import parse_query_bands, target_query


def _model() -> OpenVocabularyDepthFieldDiffusion:
    return OpenVocabularyDepthFieldDiffusion(
        OpenVocabularyDepthFieldConfig(
            spectrum_shape=(3, 16),
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
            query_spectrum=True,
            candidate_cross_attention=True,
        )
    )


def _patched_model(query_points: int = 16) -> OpenVocabularyDepthFieldDiffusion:
    return OpenVocabularyDepthFieldDiffusion(
        OpenVocabularyDepthFieldConfig(
            spectrum_shape=(3, query_points),
            depth_bins=12,
            depth_patch_size=4,
            max_candidates=3,
            d_model=16,
            n_blocks=2,
            n_heads=2,
            kernel_size=3,
            hybrid_dilations=(1, 1),
            spectrum_patch_size=2,
            spectrum_patch_stride=2,
            spectrum_encoder_blocks=1,
            spectrum_encoder_heads=2,
            query_spectrum=True,
            candidate_cross_attention=True,
        )
    )


def test_coordinate_queries_cover_full_window_and_dual_bands() -> None:
    """A fixed query length must retain variable physical wavelength support."""
    source = torch.arange(300.0, 1701.0, 5.0)
    for mode in ("coordinate_full", "coordinate_window", "coordinate_dual"):
        query = coordinate_query_wavelengths(source, points=128, mode=mode, generator=torch.Generator().manual_seed(3))
        assert query.shape == (128,)
        assert bool(torch.all(query[1:] > query[:-1]))
        assert float(query[0]) >= 299.999 and float(query[-1]) <= 1700.001
        if mode.endswith("dual"):
            assert float(torch.diff(query).max()) > 100.0


def test_256_point_inverse_query_retains_fixed_shape_and_wavelength_order() -> None:
    """Inverse-wavelength sampling keeps a sorted, fixed-size query."""
    source = torch.arange(300.0, 1701.0, 5.0)
    query = coordinate_query_wavelengths(source, points=256, mode="full")
    assert query.shape == (256,)
    assert bool(torch.all(query[1:] > query[:-1]))
    assert int((query >= 1100).sum()) == 30
    assert float(torch.diff(query).max()) > 30.0


def test_query_collator_uses_all_source_spectra_and_aligns_material_curves() -> None:
    """Query spectra and optical constants must share the same coordinates."""
    source = torch.arange(300.0, 1701.0, 5.0)
    native = source.numpy()
    catalog = MaterialCatalog(
        names=("A", "B"),
        wavelengths_nm=(native, native),
        n_values=(np.ones(len(native)), np.full(len(native), 2.0)),
        k_values=(np.zeros(len(native)), np.zeros(len(native))),
    )
    collator = OpenVocabularyDepthFieldCollator(
        wavelengths_nm=source,
        catalog=catalog,
        idx_to_token={0: "<PAD>", 1: "<MSK>", 2: "<EOS>", 3: "A_10", 4: "B_15"},
        eos_idx=2,
        pad_idx=0,
        msk_idx=1,
        channels=("R", "A", "T"),
        max_layers=4,
        max_candidates=2,
        min_query_points=16,
        max_query_points=16,
        query_sampling="coordinate_dual",
        include_source_spectrum=True,
        randomize_candidates=False,
        random_distractors=True,
        max_random_distractors=0,
        full_bank_probability=0.0,
        dz_nm=5.0,
        max_total_nm=40.0,
    )
    reflectance = (source - 300.0) / 1400.0
    spectrum = torch.stack((reflectance, 1.0 - reflectance, torch.zeros_like(reflectance)))
    batch = collator([(spectrum, torch.tensor([3, 4, 2]), 0), (spectrum, torch.tensor([4, 2]), 1)])
    assert batch["target_spectrum"].shape == (2, 16, 3)
    assert batch["source_spectrum_rat"].shape == (2, len(source), 3)
    torch.testing.assert_close(batch["source_spectrum_rat"][0], spectrum.transpose(0, 1))
    assert batch["candidate_nk"].shape == (2, 2, 16, 2)
    torch.testing.assert_close(batch["target_spectrum"][0, :, 0], (batch["wavelengths_nm"][0] - 300.0) / 1400.0)
    torch.testing.assert_close(batch["candidate_nk"][0, 0, :, 0], torch.ones(16))
    assert batch["candidate_mask"].sum(dim=1).tolist() == [2, 1]
    collator.full_bank_probability = 1.0
    full_bank = collator([(spectrum, torch.tensor([4, 2]), 2)])
    assert full_bank["candidate_mask"].sum(dim=1).tolist() == [2]


def test_query_model_attends_to_candidates_and_preserves_bank_permutation() -> None:
    """Individual candidate attention must remain local-ID equivariant."""
    torch.manual_seed(4)
    model = _model().eval()
    spectra = torch.rand(1, 3, 16)
    wavelengths = torch.linspace(400.0, 1100.0, 16).unsqueeze(0)
    curves = torch.rand(1, 3, 16, 2)
    mask = torch.tensor([[True, True, False]])
    fields = torch.tensor([[0] * 4 + [1] * 3 + [3] * 5])
    kwargs = dict(
        wavelengths_nm=wavelengths,
        candidate_mask=mask,
        incidence_angle_deg=torch.zeros(1),
        polarization_id=torch.zeros(1, dtype=torch.long),
    )
    base = model(spectra, fields, torch.tensor([0.5]), candidate_nk=curves, **kwargs)
    assert base.shape == (1, 12, 4)
    assert bool(torch.isfinite(base[..., :2]).all())
    base[..., :2].sum().backward()
    assert model.candidate_attn[0].in_proj_weight.grad is not None
    assert model.query_projection.weight.grad is not None
    with torch.no_grad():
        permuted = model(
            spectra,
            torch.where(fields == 0, 1, torch.where(fields == 1, 0, fields)),
            torch.tensor([0.5]),
            candidate_nk=curves[:, [1, 0, 2]],
            **kwargs,
        )
    torch.testing.assert_close(base.detach()[..., 0], permuted[..., 1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(base.detach()[..., 1], permuted[..., 0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(base.detach()[..., 3], permuted[..., 3], atol=1e-5, rtol=1e-5)


def test_depth_patches_keep_per_bin_logits_gradients_and_sampling() -> None:
    """Patch attention does not coarsen material predictions or loss."""
    torch.manual_seed(7)
    model = _patched_model()
    spectra = torch.rand(1, 3, 16)
    wavelengths = torch.linspace(300.0, 1700.0, 16).unsqueeze(0)
    candidate_nk = torch.rand(1, 3, 16, 2)
    candidate_mask = torch.tensor([[True, True, False]])
    fields = torch.tensor([[0, 0, 0, 1, 1, 1, 3, 3, 1, 0, 3, 3]])
    kwargs = dict(
        wavelengths_nm=wavelengths,
        candidate_nk=candidate_nk,
        candidate_mask=candidate_mask,
        incidence_angle_deg=torch.zeros(1),
        polarization_id=torch.zeros(1, dtype=torch.long),
    )
    lengths: list[int] = []
    handle = model.blocks[0].register_forward_pre_hook(lambda _module, inputs: lengths.append(inputs[0].shape[1]))
    logits = model(spectra, fields, torch.tensor([0.5]), **kwargs)
    handle.remove()
    assert lengths == [3]
    assert logits.shape == (1, 12, 4)
    assert model.convolution_receptive_field_bins == 36
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), fields.flatten()).backward()
    assert model.depth_patch_input.weight.grad is not None
    assert model.depth_patch_output.weight.grad is not None
    assert model.depth_patch_refine[2].weight.grad is not None
    with torch.no_grad():
        sampled = model.sample(spectra=spectra, steps=2, deterministic=True, **kwargs)
    assert sampled.shape == fields.shape
    assert bool((sampled != model.mask_id).all())


def test_old_checkpoint_metadata_defaults_to_unpatched_depth() -> None:
    """Existing checkpoints remain on the one-bin-per-token path."""
    metadata = _model().open_vocab_config.to_dict()
    del metadata["depth_patch_size"]
    restored = OpenVocabularyDepthFieldConfig.from_dict(metadata)
    assert restored.depth_patch_size == 1


def test_patched_model_uses_256_spectrum_points_without_changing_depth_output() -> None:
    """The wider spectrum query still produces logits for every depth bin."""
    model = _patched_model(query_points=256).eval()
    wavelengths = coordinate_query_wavelengths(torch.arange(300.0, 1701.0, 5.0), points=256, mode="full")
    tokens = model._query_spectrum_tokens(torch.rand(1, 3, 256), wavelengths.unsqueeze(0))
    assert tokens.shape == (1, 129, 16)
    with torch.no_grad():
        logits = model(
            torch.rand(1, 3, 256),
            torch.full((1, 12), model.mask_id),
            torch.tensor([1.0]),
            wavelengths_nm=wavelengths.unsqueeze(0),
            candidate_nk=torch.rand(1, 3, 256, 2),
            candidate_mask=torch.tensor([[True, True, False]]),
            incidence_angle_deg=torch.zeros(1),
            polarization_id=torch.zeros(1, dtype=torch.long),
        )
    assert logits.shape == (1, 12, 4)


def test_target_query_keeps_disjoint_gap_unconstrained() -> None:
    """Do not silently interpolate a target across an unobserved gap."""
    source = torch.cat((torch.arange(400.0, 501.0, 5.0), torch.arange(900.0, 1001.0, 5.0)))
    spectra = torch.stack((source / 1000.0, 1.0 - source / 1000.0, torch.zeros_like(source)))
    try:
        target_query(spectra, source, points=16, bands=())
    except ValueError as error:
        assert "--query-bands" in str(error)
    else:
        raise AssertionError("A wavelength gap must require explicit bands.")
    bands = parse_query_bands("400:500,900:1000")
    result, query = target_query(spectra, source, points=16, bands=bands)
    assert query.shape == (16,)
    assert float(torch.diff(query).max()) > 300.0
    torch.testing.assert_close(result[0], query / 1000.0)
    dense = interpolate_query_spectra(spectra.unsqueeze(0), source, query)
    torch.testing.assert_close(dense[0], result)
