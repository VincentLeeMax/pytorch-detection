
import torch

class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        result = {}
        for b in batch:
            for k in b:
                result.setdefault(k, [])
                result[k].append(b[k])
        result['image'] = torch.cat(result['image'])

        return result