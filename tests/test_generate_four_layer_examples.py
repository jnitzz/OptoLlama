from scripts.generate_four_layer_examples import SAMPLES, encode_samples


def test_curated_examples_have_four_families_with_two_samples_each() -> None:
    families = [sample["family"] for sample in SAMPLES]

    assert len(SAMPLES) == 8
    assert len(set(families)) == 4
    assert all(families.count(family) == 2 for family in set(families))
    assert all(len(sample["layers"]) == 4 for sample in SAMPLES)


def test_encoded_examples_have_eos_and_padding() -> None:
    layer_tokens = [layer for sample in SAMPLES for layer in sample["layers"]]
    tokens = list(dict.fromkeys(layer_tokens + ["<EOS>", "<PAD>", "<MSK>"]))
    token_to_idx = {token: index for index, token in enumerate(tokens)}

    encoded = encode_samples(token_to_idx, sequence_length=7)

    assert tuple(encoded.shape) == (8, 7)
    assert (encoded[:, 4] == token_to_idx["<EOS>"]).all()
    assert (encoded[:, 5:] == token_to_idx["<PAD>"]).all()
