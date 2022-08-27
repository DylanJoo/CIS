import os
import argparse
import random
import collections
import json
from utils import load_queries, load_runs, load_collections, doc_pool_random_sampling, load_topics, normalized

parser = argparse.ArgumentParser()
parser.add_argument("--topic", default="data/canard/train.queries.jsonl", type=str)
parser.add_argument("--run", default="spr/runs/train.canard.answer.top200.trec", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--topk_pool", type=int, default=200)
parser.add_argument("--topk_positive", type=int, default=None)
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--window_size", type=int, default=0)
parser.add_argument("--collections", type=str, default="data/trec-car+marco-psg/")
parser.add_argument("--multiview", action='store_true', default=False)
args = parser.parse_args()


queries = load_topics(args.topic)
run = load_runs(args.run)
passages = load_collections(dir=args.collections)

# set seed
random.seed(123)
# count = collections.defaultdict(list)
fout = open(args.output, 'w')

with open(args.topic) as topic:
    for i, line in enumerate(topic):
        query_dict = json.loads(line.strip())
        qid = query_dict['id']
        ranklist = [docid for docid in run[qid][:args.topk_pool]]

        # heuristic positive negative boundary
        positive_pool = ranklist[:args.topk_positive]
        negative_pool = ranklist[args.topk_positive:args.topk_pool]

        # sampling positives and negatives
        psg_ids_pos = doc_pool_random_sampling(positive_pool, args.n)
        psg_ids_neg = doc_pool_random_sampling(negative_pool, args.n)
        
        q = normalized(query_dict['utterance'])
        c_t = "|".join(query_dict['history_topic'])
        c_u = query_dict['history_utterances'][-args.window_size:]
        c = normalized("|".join([c_t] + c_u))

        for j, (psg_id_pos, psg_id_neg) in enumerate(zip(psg_ids_pos, psg_ids_neg)):
            d_pos = normalized(passages[psg_id_pos])
            d_neg = normalized(passages[psg_id_neg])
            fout.write(f"Query: {q} Context: {c} Document: {d_pos} Relevant:\ttrue\n")
            fout.write(f"Query: {q} Context: {c} Document: {d_neg} Relevant:\tfalse\n")

        # [TODO] Considering the mixing way, so far, cannot activated
        # if args.multiview:
        #     fout.write(f"Query: {q} Context: {c} Rewrite:\t{query_dict['rewrite']}\n")

    if i % 10000 == 0:
        print(f"{i} convir queries finished...")

fout.close()

print("DONE")

