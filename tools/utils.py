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

def load_runs(path, output_score=False): # support .trec file only
    run_dict = collections.defaultdict(list)
    sorted_run_dict = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            qid, Q0, docid, rank, rel_score, pyserini = line.strip().split()
            run_dict[qid] += [(docid, rel_score)]

    # sort, just in case
    for (qid, doc_id_ranklist) in run_dict.items():
        sorted_ranklist = sorted(doc_id_ranklist, key=lambda x: x[1], reverse=True) # score with descending order
        if output_score:
            sorted_run_dict[qid] = [(docid, rel_score) for docid, rel_score in sorted_ranklist]
        else:
            sorted_run_dict[qid] = [docid for docid, _ in sorted_ranklist]

    return sorted_run_dict

def load_collections(path=None, dir=None, candidate_set=None):
    collection_dict = collections.defaultdict(str)

    if dir: # load if there are many jsonl files
        files = [os.path.join(dir, f) for f in os.listdir(dir) if ".json" in f]
    else:
        files = [path]

    for file in files:
        print(f"Loading from collection {file}...")
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                example = json.loads(line.strip())
                if candidate_set:
                    if example['id'] in candidate_set:
                        collection_dict[example['id']] = example['contents'].strip()
                        candidate_set.remove(example['id'])
                    if len(candidate_set) == 0:
                        break
                else:
                    collection_dict[example['id']] = example['contents'].strip()

                if i % 1000000 == 1:
                    print(f" # documents...{i}")

    print("DONE")
    return collection_dict

def filter_rel_collections(corpus, runs, output='rel_collections.jsonl'):
    collection_dict = {}
    rel_doc_set = set()
    for i in range(len(runs)):
        for k, rank_list in runs[i].items():
            rel_doc_set.update([docid for (docid, _) in rank_list])

    fout = open(output, 'w')

    while len(rel_doc_set) > 0:
        docid = rel_doc_set.pop()
        doctext = corpus[docid]

        if output is not None:
            fout.write(
                    json.dumps({'id': docid, 'contents': doctext.replace("\n", "")})+'\n'
            )
        else:
            collection_dict[docid] = doctext.replace("\n", "")

        if len(rel_doc_set) % 10000 == 1:
            print(f'{len(rel_doc_set)} passages remain unfound.')

    print("dONE")
    if output is None:
        return collection_dict





