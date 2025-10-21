import pandas as pd
import torch
import torch.utils.data

from utils import save_JSONPICKLE_NEW, load_JSONPICKLE_NEW

def parse_tokens(path: str) -> pd.DataFrame:
    tokens = load_JSONPICKLE_NEW(path)

    split = list(map(lambda x: x.split('_'), tokens))
    # introduce special tokens
    split.append(['<EOS>', '0'])
    split.append(['<PAD>', '0'])
    split.append(['<MSK>', '0'])

    df = pd.DataFrame(split, columns=['material', 'thickness'])
    df['thickness'] = df['thickness'].astype(float)
    df['thickness'] /= df['thickness'].max()
    ids = (df['material'] != df['material'].shift(1)).cumsum() - 1
    
    return pd.concat([df, pd.get_dummies(ids).astype(float)], axis=1)


def pad_batch(batch: list[torch.Tensor], max_stack_depth: int, eos: int, pad: int):
    spectra, stacks, indices = [], [], []

    for item in batch:
        # support (spectrum, stack) or (spectrum, stack, index)
        if len(item) == 3:
            spectrum, stack, idx = item
            indices.append(idx)
        else:
            spectrum, stack = item
        # pass spectrum through
        spectra.append(spectrum)

        # fix stacks
        stack_depth = stack.shape[0]
        padding = torch.zeros(max_stack_depth - stack_depth + 1, dtype=torch.long)
        padding[:] = pad
        padding[0] = eos
        stacks.append(torch.cat((stack, padding,)))
    out_spectra = torch.stack(spectra, dim=0)
    out_stacks  = torch.stack(stacks,  dim=0)
    if indices:
        return out_spectra, out_stacks, torch.as_tensor(indices, dtype=torch.long)
    else:
        return out_spectra, out_stacks    
    # return torch.stack(spectra, dim=0), torch.stack(stacks, dim=0)


class SpectraDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[str], tokens: torch.Tensor, device: str):
        super().__init__()
        
        if isinstance(paths, str):
            paths = [paths]

        # Load and concatenate spectra and stacks from all paths
        all_spectra = []
        all_stacks = []

        for path in paths:
            spectra, stacks = torch.load(path, weights_only=False)
            all_spectra.extend(spectra)
            all_stacks.extend(stacks)

        self.spectra = all_spectra
        self.stacks = all_stacks
        self.maximum_depth = max(stack.shape[0] for stack in self.stacks)

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor, int):
        return (
            self.spectra[index].reshape(3, -1).T.to(torch.float32),
            self.stacks[index],
            index
        )

    def get_maximum_depth(self) -> int:
        return self.maximum_depth
