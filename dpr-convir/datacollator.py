from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import (
        PaddingStrategy,
        PreTrainedTokenizerBase
)
# from prepare_dataset import prepare_colbert_triplet

def rename_inputs(x, prefix=""):
    keys = list(x.keys())
    for k in keys:
        x[f'{prefix}{k}'] = x.pop(k)
    return 0

def stack_inputs(x, rep=2):
    for k in x:
        x.update({k: x[k]*2})
    return 0

def merge_inputs(qx, dx):
    for k in dx:
        qx.update({k: dx[k]})
    return qx

@dataclass
class IRTripletCollator:
    tokenizer: PreTrainedTokenizerBase
    query_maxlen: Optional[int] = None
    doc_maxlen: Optional[int] = None
    return_tensors: str = "pt"
    in_batch_negative: Optional[bool] = False
    # pad_to_multiple_of: Optional[int] = None
    # padding: Union[bool, str, PaddingStrategy] = True
    # truncation: Union[bool, str] = True
    # max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        q_texts = [f"[CLS] [Q] {text['query']}" + "[MASK]" * self.query_maxlen for text in features]
        d1_texts = [f"[CLS] [D] {text['pos_passage']}" for text in features]
        d0_texts = [f"[CLS] [D] {text['neg_passage']}" for text in features]

        # ColBert setting 
        ## Query tokenization
        q_inputs = self.tokenizer(
            q_texts,
            max_length=self.query_maxlen,
            truncation=True,
            add_special_tokens=False,
        )
        ## Document tokenization
        ### positive and negative
        d_inputs = self.tokenizer(
            d1_texts + d0_texts, # positive doc + negative doc
            max_length=self.doc_maxlen,
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        )

        ## post-processing
        ### renameing (for model)
        rename_inputs(q_inputs, 'q_')
        rename_inputs(d_inputs, 'd_')

        ### stack query input (for pairwise loss)
        if not self.in_batch_negative:
            stack_inputs(q_inputs, rep=2)
        ### merge triplet input into one
        inputs = merge_inputs(q_inputs, d_inputs)

        return inputs.convert_to_tensors(self.return_tensors)

