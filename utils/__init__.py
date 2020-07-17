import random
import numpy
import torch
import torch.nn as nn


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeviceAwareModule(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device