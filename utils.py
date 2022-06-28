import json
import argparse
import random
import collections
import os

def load_queries(path):
    query_dict = {}
    with open(path, 'r') as f:
        for line in f:
            qid, qtext = line.strip().split('\t')
            query_dict[qid] = qtext.strip()

    return query_dict

def load_runs(path): # support .trec file only
    run_dict = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            qid, Q0, docid, rank, rel_score, pyserini = line.strip().split()
            run_dict[qid] += [(docid, rank)]

    # sort, just in case
    for qid in run_dict:
        run_dict[qid].sort(key=lambda x: x[1], reverse=True)

    return run_dict

def load_collections(dir):
    collection_dict = collections.defaultdict(str)
    files = [f for f in os.listdir(dir) if ".json" in f]

    for file in files:
        print(f"Loading from collection {file}...")
        with open(os.path.join(dir, file), 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                collection_dict[example['id']] = example['contents'].strip()

    return collection_dict
