import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class PointwiseDataCollatorForT5:
    tokenizer: PreTrainedTokenizerBase
    query_maxlen: Optional[int] = None
    doc_maxlen: Optional[int] = None
    return_tensors: Optional[str] = None
    query_source: Optional[str] = "automatic_rewritten"
    is_train: bool = True
    # context_maxlen: Optional[int] = None
    # utterance_maxlen: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        q_texts = [text[f'{self.query_source}'] for text in features]
        d_texts = [text['passage'] for text in features] 

        ## Rewrite text + Document tokenization
        inputs = self.tokenizer(
                [f"Query: {q} Document: {d} Relevant:" for q, d in zip(q_texts, d_texts)],
                max_length=self.query_maxlen+self.doc_maxlen,
                padding=True,
                truncation=True,
                return_tensors=self.return_tensors
        )

        if self.is_train:
            ## target text
            targets = self.tokenizer(
                    [text['label'] for text in features],
                    truncation=True,
                    return_tensors=self.return_tensors
            )
            # labels
            inputs['labels'] = targets.input_ids

        return inputs

@dataclass
class PointwiseConvDataCollatorForT5:
    tokenizer: PreTrainedTokenizerBase
    query_maxlen: Optional[int] = None
    doc_maxlen: Optional[int] = None
    return_tensors: Optional[str] = None
    num_history: Optional[int] = 1
    num_history_utterances: Optional[int] = None
    is_train: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        u_texts = [text['utterance'] for text in features]
        d_texts = [text['passage'] for text in features] 
        c_texts = []

        for i, text in enumerate(features):
            c_text = text['context'].split('|') # odds are queries, even are docs
            # c_texts.append('|'.join(c_text[-self.num_history_utterances:]))
            if self.num_history_utterances is not None:
                cq_text = [c for i, c in enumerate(c_text) \
                        if i % 2 == 0][-self.num_history_utterances]
                c_texts.append('|'.join(cq_text))
            else:
                c_texts.append('|'.join(c_text[-self.num_history*2:]))

        ## Document tokenization
        d_inputs = self.tokenizer(
                [f"Document: {d} Relevant:" for d in d_texts],
                max_length=self.doc_maxlen,
                padding="longest",
                truncation="longest_first",
                return_tensors=self.return_tensors
        )

        ## Utterance text + Context text + (truncated) Document listOfTokens
        inputs = self.tokenizer(
                [f"Query: {u} Context: {c}" for (u, c) in zip(u_texts, c_texts)],
                max_length=self.query_maxlen,
                padding="max_length",
                truncation="longest_first",
                add_special_tokens=False,
                return_tensors=self.return_tensors
        )

        if self.is_train:
            ## target text
            targets = self.tokenizer(
                    [text['label'] for text in features],
                    truncation=True,
                    return_tensors=self.return_tensors
            )
            # labels
            inputs['labels'] = targets.input_ids

        # Concatentate
        for k in ['input_ids', 'attention_mask']:
            inputs[k] = torch.cat((inputs[k], d_inputs[k]), 1)


        return inputs
