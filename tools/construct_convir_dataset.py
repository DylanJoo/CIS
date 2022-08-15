import os
import argparse
import random
import collections
import json
from utils import load_queries, load_runs, load_collections, doc_pool_random_sampling, load_topics, normalized

parser = argparse.ArgumentParser()
parser.add_argument("--topic", default="data/canard/train.queries.jsonl", type=str)
parser.add_argument("--run_target", default="spr/runs/train.canard.answer.top200.trec", type=str)
parser.add_argument("--run_reference", default="spr/runs/train.canard.rewrite.top200.trec", type=str)
parser.add_argument("--convir_dataset", default="convir_data/canard_convir.train.cqe.jsonl", type=str)
parser.add_argument("-k", "--topk_pool", type=int, default=200)
parser.add_argument("-k_pos", "--topk_positive", type=int, default=None)
parser.add_argument("-n", "--n_examples", type=int, default=100)
parser.add_argument("--discard_history_responses", action='store_true', default=False)
parser.add_argument("--window_size", type=int, default=0)
parser.add_argument("--version", type=str, default="top3")
parser.add_argument("-collections", "--collections", type=str, default="data/trec-car+marco-psg/")
parser.add_argument("--multiview", action='store_true', default=False)
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
random.seed(123)
# count = collections.defaultdict(list)
fout = open(args.convir_dataset, 'w')

with open(args.topic) as topic:
# for i, query_dict in enumerate(convir_queries.items()):
    for i, line in enumerate(topic):
        query_dict = json.loads(line.strip())
        qid = query_dict['id']
        ranklist_teacher = [docid for docid in run_teacher[qid][:args.topk_pool]]
        ranklist_student = [docid for docid in run_student[qid][:args.topk_pool]]

        # get the positive pool and negative pool wrt query
        if args.version == 'overlapped':
            positive_pool = \
                    [docid for docid in ranklist_teacher if docid in ranklist_student][:args.topk_positive]
            negative_pool = [docid for docid in ranklist_student if docid not in ranklist_teacher]

            # corncer case I: OVERLAPPED < 3
            if len(positive_pool) < args.topk_positive:
                positive_pool = positive_pool + ranklist_teacher[:args.topk_positive]
            # corncer case II: OVERLAPPED > 197
            if len(negative_pool) < 30: # 3 (positive) * 33 (negative)
                negative_pool = negative_pool + ranklist_student[-30:]

        if args.version == 'top3':
            positive_pool = ranklist_student[:args.topk_positive]
            negative_pool = ranklist_student[args.topk_positive:args.topk_pool]


        # sampling positives and negatives
        psg_ids_pos = doc_pool_random_sampling(positive_pool, args.n_examples)
        psg_ids_neg = doc_pool_random_sampling(negative_pool, args.n_examples)
        
        q = normalized(query_dict['utterance'])
        c_t = "|".join(query_dict['history_topic'])
        c_u = query_dict['history_utterances'][-args.window_size:]
        c_r = query_dict['history_responses'][-args.window_size:]

        if args.discard_history_responses:
            c = normalized("|".join([c_t] + c_u))
        else:
            c = normalized("|".join([c_t] + [f"{u}|{r}" for u, r in zip(c_u, c_r)]))

        for j, (psg_id_pos, psg_id_neg) in enumerate(zip(psg_ids_pos, psg_ids_neg)):
            d_pos = normalized(passages[psg_id_pos])
            d_neg = normalized(passages[psg_id_neg])
            fout.write(f"Query: {q} Context: {c} Document: {d_pos} Relevant:\ttrue\n")
            fout.write(f"Query: {q} Context: {c} Document: {d_neg} Relevant:\tfalse\n")
            if j % 4 == 0 and args.multiview:
                fout.write(f"Query: {q} Context: {c} Rewrite:\t{query_dict['rewrite']}\n")
            # else:
            #     fout.write(f"{q}\t{c}\t{d_pos}\ttrue\n")
            #     fout.write(f"{q}\t{c}\t{d_neg}\tfalse\n")

    if i % 10000 == 0:
        print(f"{i} convir queries finished...")

fout.close()

print("DONE")

