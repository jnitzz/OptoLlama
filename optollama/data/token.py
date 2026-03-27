import torch

import optollama.utils

PAD_TOKEN = "<PAD>"
MSK_TOKEN = "<MSK>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = {
    PAD_TOKEN,
    MSK_TOKEN,
    EOS_TOKEN
}


def material_name(token: str) -> str:
    """
    Extract a base material name from a token string.

    Robust to common formats where thickness/parameters are appended, e.g.
    ``"SiO2_120"``, ``"TiN(30nm)"``, etc. Adjust the split logic if your
    token format differs.

    Args
    ----
    token : str
        A token string such as ``"SiO2_120"`` or ``"TiN"``.

    Returns
    -------
    str
        The base material name (everything before the first ``"_"``).
    """
    return token.split("_", 1)[0]


def make_material_groups(tokens: list[str], token_to_idx: dict[str, int]) -> dict[str, torch.Tensor]:
    """
    Build predefined material-group token-id sets from the vocabulary.

    Partitions all non-special tokens into three groups:

    - ``"metals"`` — Ag, Al, TiN
    - ``"semiconductors"`` — Ge, ITO, Si, ZnO, ZnS, ZnSe
    - ``"dielectrics"`` — all remaining non-special tokens

    Args
    ----
    tokens : list[str]
        Full list of token strings in the vocabulary (including special tokens).
    token_to_idx : dict[str, int]
        Mapping from token string to integer id.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys ``"metals"``, ``"semiconductors"``, and
        ``"dielectrics"``, each mapping to a 1-D long tensor of token ids
        belonging to that group.
    """
    metals = {"Ag", "Al", "TiN"}
    semis = {"Ge", "ITO", "Si", "ZnO", "ZnS", "ZnSe"}

    metal_tokens, semiconductor_tokens, other_tokens = [], [], []
    
    for t in tokens:
        if t in SPECIAL_TOKENS:
            continue

        b = material_name(t)
        
        if b in metals:
            metal_tokens.append(t)
        elif b in semis:
            semiconductor_tokens.append(t)
        else:
            other_tokens.append(t)

    return {
        "metals": torch.tensor([token_to_idx[token] for token in metal_tokens], dtype=torch.long),
        "semiconductors": torch.tensor([token_to_idx[token] for token in semiconductor_tokens], dtype=torch.long),
        "dielectrics": torch.tensor([token_to_idx[token] for token in other_tokens], dtype=torch.long),
    }


def make_material_token_ids(token_to_idx: dict[str, int]) -> dict[str, list[int]]:
    """
    Build a mapping from base material names to lists of token ids.

    Iterates over the full vocabulary and groups token ids by their base
    material name (e.g. all ``"SiO2_*"`` tokens are grouped under ``"SiO2"``).
    Special tokens (PAD, EOS, MSK) are excluded.

    Args
    ----
    token_to_idx : dict[str, int]
        Full vocabulary mapping from token string to integer id.

    Returns
    -------
    dict[str, list[int]]
        Mapping from base material name to a list of corresponding token ids.
    """
    material_ids = {}

    for token, token_id in token_to_idx.items():
        if token in SPECIAL_TOKENS:
            continue

        material = material_name(token)
        material_ids.setdefault(material, []).append(token_id)

    return material_ids


def token_ids_of(items: list, token_to_idx: dict[str, int], base_to_ids: dict[str, list[int]]) -> torch.Tensor:
    """
    Expand a mixed list of material names or token ids into a tensor of token ids.

    Each entry in ``items`` is resolved as follows:

    - ``int`` → used directly as a token id.
    - ``str`` → looked up as an exact token first (e.g. ``"SiO2_120"``);
      if not found, treated as a base material name and expanded to all
      matching token ids (e.g. ``"SiO2"`` → all ``"SiO2_*"`` ids).

    Args
    ----
    items : list
        Mixed list of ``int`` token ids or ``str`` token/material names.
    token_to_idx : dict[str, int]
        Full vocabulary mapping from token string to integer id.
    base_to_ids : dict[str, list[int]]
        Mapping from base material name to a list of token ids,
        as produced by :func:`make_material_token_ids`.

    Returns
    -------
    torch.Tensor
        1-D long tensor of unique resolved token ids. Empty tensor if
        ``items`` is empty.

    Raises
    ------
    ValueError
        If a string entry cannot be resolved as an exact token or base material.
    TypeError
        If an entry is neither ``int`` nor ``str``.
    """
    ids = []
    for x in items:
        if isinstance(x, int):
            ids.append(int(x))
        elif isinstance(x, str):
            s = x.strip()
            if s in token_to_idx:
                ids.append(int(token_to_idx[s]))
            elif s in base_to_ids:
                ids.extend(base_to_ids[s])
            else:
                raise ValueError(
                    f"Unknown token/material in filter list: {s!r}. Use exact token like 'SiO2_120' or base material like 'SiO2'."
                )
        else:
            raise TypeError(f"Filter entries must be int or str, got: {type(x)}")

    if not ids:
        return torch.empty((0,), dtype=torch.long)

    return torch.unique(torch.tensor(ids, dtype=torch.long))


def init_tokens(path: str) -> tuple[list[str], dict[str, int], dict[int, str], str, str, str, int, int, int]:
    """
    Return the list of tokens, two dicts with idx/tokens, the special token strings and their respective id.

    Args
    ----
    path: str
        Path to the tokens.json file
    
    Returns
    -------
    dict
        Combination of mappings from tokens to ids and vice versa.
    """
    tokens = optollama.utils.load_as_json(path)
    token_to_idx = {tk: i for i, tk in enumerate(tokens)}
    eos_idx = token_to_idx[EOS_TOKEN]
    pad_idx = token_to_idx[PAD_TOKEN]
    msk_idx = token_to_idx[MSK_TOKEN]
    idx_to_token = {i: tk for i, tk in enumerate(token_to_idx)}

    return tokens, token_to_idx, idx_to_token, EOS_TOKEN, PAD_TOKEN, MSK_TOKEN, eos_idx, pad_idx, msk_idx
