import argparse
import collections
from models import monoT5
from datasets import Dataset
from torch.utils.data import DataLoader
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
from datacollator import PointwiseDataCollatorForT5, PointwiseConvDataCollatorForT5
from transformers import AutoTokenizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_pair", type=str, required=True,)
    parser.add_argument("--input_trec", type=str, required=True)
    parser.add_argument("--output_trec", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_q_seq_length", type=int, default=128)
    parser.add_argument("--max_p_seq_length", type=int, default=384)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--prefix", type=str, default='monoT5')
    args = parser.parse_args()

    fout = open(args.output_trec, 'w')

    # load model
    model = monoT5.from_pretrained(args.model_name_or_path)
    model.set_tokenizer()
    model.set_targets(['true', 'false'])
    model.eval()

    # load data
    data = Dataset.from_json(args.jsonl_pair)

    # data loader
    if args.prefix == 'monoT5':
        datacollator = PointwiseDataCollatorForT5(
                tokenizer=model.tokenizer,
                query_maxlen=args.max_q_seq_length,
                doc_maxlen=args.max_p_seq_length,
                return_tensors='pt',
                query_source='automatic_rewritten',
                is_train=False
        )

    if args.prefix == 'conv.monoT5':
        datacollator = PointwiseConvDataCollatorForT5(
                tokenizer=model.tokenizer,
                query_maxlen=args.max_q_seq_length,
                doc_maxlen=args.max_p_seq_length,
                return_tensors='pt',
                num_history=3,
                is_train=False
        )

    dataloader = DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=datacollator
    )

    model.to(f'cuda:{args.gpu}')
    model.eval()

    score_list = []
    # run prediction
    for b, batch_input in enumerate(dataloader):
        output = model.predict(batch_input)

        true_prob = output[:, 0]
        false_prob = output[:, 1]

        score_list += true_prob.tolist()

        if b % 1000 == 0:
            print(true_prob)
            print(f"{b} qp pair inferencing")


    ranking_list = collections.defaultdict(list)
    with open(args.input_trec, 'r') as f:
        for line, t_prob in zip(f, score_list):
            qid, Q0 ,docid, rank, _, prefix = line.strip().split()
            ranking_list[qid].append((docid, t_prob))
            
    with open(args.output_trec, 'w') as f:
        for i, (qid, docid_score_list) in enumerate(ranking_list.items()):

            docid_score_list_sorted = sorted(docid_score_list, key=lambda x: x[1], reverse=True)
            for idx, (docid, t_prob) in enumerate(docid_score_list_sorted):
                example = f'{qid} Q0 {docid} {str(idx+1)} {t_prob} {args.prefix}\n'
                fout.write(example)
