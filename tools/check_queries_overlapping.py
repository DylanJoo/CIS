import os
import argparse
import random
import collections
import numpy as np
import json
from utils import load_queries, load_runs, load_collections, doc_pool_random_sampling, load_topics, normalized

parser = argparse.ArgumentParser()
parser.add_argument("--topic", default="data/canard/train.queries.jsonl", type=str)
parser.add_argument("--run_target", default="spr/runs/train.canard.answer.top200.trec", type=str)
parser.add_argument("--run_reference", default="spr/runs/train.canard.rewrite.top200.trec", type=str)
parser.add_argument("--convir_dataset_stats", type=str)
parser.add_argument("-k", "--topk_pool", type=int, default=200)
parser.add_argument("-k_pos", "--topk_positive", type=int, default=None)
parser.add_argument("--n_examples", type=int, default=5)
parser.add_argument("-collections", "--collections", type=str, default="data/trec-car+marco-psg/")
args = parser.parse_args()


# load query
convir_queries = load_topics(args.topic)
# load runs (reranked)
run_student = load_runs(args.run_target)
run_teacher = load_runs(args.run_reference)
assert len(run_student) == len(run_teacher), "Inconsistent number of queries"
# load documents
if os.path.isdir(args.collections):
    passages = load_collections(dir=args.collections)
else:
    passages = load_collections(path=args.collections)

# set seed
random.seed(777)
# count = collections.defaultdict(list)
counter_overlapped = []
info_overlapped = []

fout = open(args.convir_dataset_stats, 'w')

with open(args.topic) as topic:
# for i, query_dict in enumerate(convir_queries.items()):
    for i, line in enumerate(topic):
        query_dict = json.loads(line.strip())
        qid = query_dict['id']
        ranklist_teacher = [docid for docid in run_teacher[qid][:args.topk_pool]]
        ranklist_student = [docid for docid in run_student[qid][:args.topk_pool]]

        # get the positive pool and negative pool wrt query 
        # (Noted that the following two list is order-sensitive.)
        overlapped = [docid for docid in ranklist_teacher if docid in ranklist_student]
        difference = [docid for docid in ranklist_student if docid not in ranklist_teacher]

        positive_pool = overlapped[:args.topk_pool]
        negative_pool = difference

        # CORNCER CASE I: OVERLAPPED < 3
        if len(positive_pool) < args.topk_pool:
            positive_pool = ranklist_teacher[:args.topk_positive]
        # CORNCER CASE II: OVERLAPPED > 197
        if len(negative_pool) < 33.3333: # 3 (positive) * 33 (negative)
            negative_pool = ranklist_student[-40:]

        # sampling positives and negatives
        psg_ids_pos = doc_pool_random_sampling(positive_pool, args.n_examples)
        psg_ids_neg = doc_pool_random_sampling(negative_pool, args.n_examples)
        
        q = normalized(query_dict['utterance'])
        c_t = "|".join(query_dict['history_topic'])
        c_u = query_dict['history_utterances']
        c_r = query_dict['history_responses']
        c = normalized("|".join([c_t] + [f"{u}|{r}" for u, r in zip(c_u, c_r)]))

        counter_overlapped.append(len(overlapped))
        info_overlapped.append({
            "Query": query_dict['rewrite'], 
            "Answer": query_dict['answer'], 
            "First P+": passages[positive_pool[0]],
            "Last P-": passages[negative_pool[-1]],
        })
        # fout.write(
        #         f"Query: {query_dict['rewrite']}\
        #         \n\n# Intersections: {len(overlapped)}\
        #         \nFirst P+: {passages[positive_pool[0]]}\
        #         \nLast P-: {passages[negative_pool[-1]]}\n"
        # )

    sorted_index = np.argsort(counter_overlapped)
    for idx in sorted_index:
        c = counter_overlapped[idx]
        i_dict = info_overlapped[idx]
        fout.write(f"# Num. Overlapped: {c:.<5}")
        fout.write(f"- Query: {i_dict['Query']}\n")
        fout.write(f"- Answer: {i_dict['Answer']}\n")
        fout.write(f"- First P+: {i_dict['First P+']}\n")
        fout.write(f"- Last P-: {i_dict['Last P-']}\n\n")

    if i % 10000 == 0:
        print(f"{i} convir queries finished...")

print("DONE")

