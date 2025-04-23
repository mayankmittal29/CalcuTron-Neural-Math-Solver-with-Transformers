import torch
from torch.autograd import Variable

def create_vocab(operations=['+', '-']):
    """Create vocabulary dictionaries mapping between characters and indices."""
    char_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    for i in range(10):
        char_to_idx[str(i)] = i + 3
    for op in operations:
        char_to_idx[op] = len(char_to_idx)
    
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char, len(char_to_idx)

class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def subsequent_mask(size):
    """Mask out subsequent positions."""
    import numpy as np
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0