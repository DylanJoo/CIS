import argparse
import random
import collections
import json
from utils import load_queries, load_runs, load_collections

parser = argparse.ArgumentParser()
parser.add_argument("--query", default="data/canard/train.rewrite.tsv", type=str)
parser.add_argument("--run_target", default="spr/runs/train.canard.answer.top200.trec", type=str)
parser.add_argument("--run_reference", default="spr/runs/train.canard.rewrite.top200.trec", type=str)
parser.add_argument("--convir_dataset", default="convir_data/canard_convir.train.cqe.jsonl", type=str)
# parser.add_argument("--negative_pool", type=str, default="rewrite-answer")
# parser.add_argument("--hn_first", action='store_true', default=False)
# parser.add_argument("--hn_rand", action='store_true', default=False)
parser.add_argument("-triplet", "--triplet", action='store_true', default=False)
parser.add_argument("-k", "--topk_pool", type=int, default=200)
parser.add_argument("-k_pos", "--topk_positive", type=int, default=None)
parser.add_argument("-n", "--n_triplets", type=int, default=100)
parser.add_argument("-collections", "--collections_dir", type=str, default="data/trec-car+marco-psg/")
args = parser.parse_args()


# load query
convir_queries = load_queries(args.query)
# load runs (reranked)
run_student = load_runs(args.run_target)
run_teacher = load_runs(args.run_reference)
assert len(run_student) == len(run_teacher), "Inconsistent number of queries"
# load documents
passages = load_collections(
        path=None,
        dir=args.collections_dir
)


# set seed
random.seed(777)
# count = collections.defaultdict(list)
fout = open(args.convir_dataset, 'w')

for i, (qid, qtext) in enumerate(convir_queries.items()):
    ranklist_teacher = [docid for docid in run_teacher[qid][:args.topk_pool]]
    ranklist_student = [docid for docid in run_student[qid][:args.topk_pool]]

    # positive_pool = [docid for docid in ranklist_student if docid in ranklist_teacher]
    # positive_pool = [docid for docid in ranklist_student if docid in ranklist_teacher]

    positive_pool = ranklist_student[:args.topk_positive]
    negative_pool = ranklist_student[args.topk_positive:args.topk_pool]

    try:
        psg_ids_pos = random.sample(positive_pool, args.n_triplets)
    except:
        psg_ids_pos = random.choices(positive_pool, k=args.n_triplets)

    try:
        psg_ids_neg = random.sample(negative_pool, args.n_triplets)
    except:
        psg_ids_neg = random.choices(negative_pool, args.n_triplets)


    for (psg_id_pos, psg_id_neg) in zip(psg_ids_pos, psg_ids_neg):
        # f writer
        # fout.write(f'{qtext}\t{passages[psg_id_pos]}\t1\n')
        # fout.write(f'{qtext}\t{passages[psg_id_neg]}\t0\n')

        # jsonl writer
        if args.triplet:
            fout.write(json.dumps({
                "query": qtext, 
                "pos_passage": passages[psg_id_pos],
                "neg_passage": passages[psg_id_neg],
            })+'\n')
        else:
            fout.write(json.dumps({
                "query": qtext, 
                "passage": passages[psg_id_pos],
                "label": 1
            })+'\n')
            fout.write(json.dumps({
                "query": qtext, 
                "passage": passages[psg_id_neg],
                "label": 0
            })+'\n')

    if i % 10000 == 0:
        print(f"{i} convir exampled finished...")

print("DONE")

