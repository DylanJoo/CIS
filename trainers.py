"""
Customized trainer for setnece highlight
"""
from transformers import Trainer
import time
import json
import collections
import multiprocessing

class AlbertTrainer(Trainer):

    def inference(self, eval_dataset=None):
        for b, batch in enumerate(eval_dataset):
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            output = self.model.inference(batch)
            logit = output['logit']



