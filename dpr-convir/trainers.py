"""
Customized trainer for setnece highlight
"""
# import time
# import json
# import collections
# import multiprocessing
import torch
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer_utils import has_length

_is_torch_generator_available = False


