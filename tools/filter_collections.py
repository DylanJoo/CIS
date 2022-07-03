import os
import argparse
import random
import collections
from utils import load_runs, load_collections, filter_rel_collections

parser = argparse.ArgumentParser()
parser.add_argument("--runs", action='append', default=[
    'spr/runs/train.canard.rewrite.top1000.trec', 'spr/runs/train.canard.answer.top1000.trec'
])
parser.add_argument("--collections_dir", type=str, default="data/trec-car+marco-psg/")
args = parser.parse_args()


runs = {}
for i in range(len(args.runs)):
    runs[i] = load_runs(args.runs[i])

passages = load_collections(args.collections_dir)

filter_rel_collections(
        collections=passages, 
        runs=runs, 
        output=os.path.join(args.collections_dir, 'rel_collections.jsonl')
)

print("DONE")
