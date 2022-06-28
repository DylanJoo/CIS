import argparse
import random
import collections
from utils import load_queries, load_runs, load_collections

parser = argparse.ArgumentParser()
parser.add_argument("--query", default="data/canard/train.rewrite.tsv", type=str)
parser.add_argument("--run_target", default="spr/runs/train.canard.answer.top200.trec", type=str)
parser.add_argument("--run_reference", default="spr/runs/train.canard.rewrite.top200.trec", type=str)
parser.add_argument("--canard_ir", default="convir_data/canard_ir.train.rewrite-answer.tsv", type=str)
# parser.add_argument("--negative_pool", type=str, default="rewrite-answer")
parser.add_argument("--hn_first", action='store_true', default=False)
parser.add_argument("--hn_rand", action='store_true', default=False)
parser.add_argument("--n_triplets", type=int, default=3)
parser.add_argument("--collections_dir", type=str, default="data/trec-car+marco-psg/")
args = parser.parse_args()



queries = load_queries(args.query)
target = load_runs(args.run_target)
reference = load_runs(args.run_reference)
assert len(target) == len(reference), "Inconsistent number of queries"
passages = load_collections(args.collections_dir)

fout = open(args.canard_ir, 'w')
args.hn_first = True

for qid, qtext in queries:

    target_set = [docid for (docid, rank) in target[qid]]
    mismatched = [docid for (docid, rank) in reference[qid] if docid not in target_set]
    matched = [docid for (docid, rank) in reference[qid] if docid in target_set]

    # positive document
    docid_pos_list = (matched + target_set)[:args.n_triplets]

    # negative document
    if args.hn_first:
        docid_neg_list = (mismatched + qe)[0]
        try:
        except:
            print(f"Query hard negative passage unfounded: {qtext}")
            docid_neg = random.sample(target_set, 1)[0]

    if args.hn_rand:
        docid_neg = random.sample(target_set, 1)

    fout.write(f'{qtext}\t{passages[docid_pos]}\n')
    fout.write(f'{qtext}\t{passages[docid_neg]}\n')

print("DONE")

