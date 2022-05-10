"""
Customized trainer for setnece highlight
"""
from transformers import Trainer
import time
import json
import collections
import multiprocessing

class AlbertTrainer(Trainer):
    f"""
    The optimization process is identical the siamese networks, 
    """

    def inference(self):
        pass

