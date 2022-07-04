import os
import collections
import re
import argparse
import json
from utils import load_runs, load_queries, load_collections

parser = argparse.ArgumentParser()
parser.add_argument("-run", "--run", type=str, required=False,)
parser.add_argument("-corpus", "--corpus", type=str, required=True,)
parser.add_argument("-d", "--doc_level", action="store_true", default=False,)
parser.add_argument("-k", "--top_k", type=int, default=1000,)
parser.add_argument("-q", "--queries", type=str, required=True,)
parser.add_argument("-q_index", "--queries_index", type=str, required=False)
parser.add_argument("--output_text_pair", type=str, required=True,)
parser.add_argument("--output_id_pair", type=str, required=True,)
args = parser.parse_args()

# 
runs = load_runs(path=args.run)
queries = load_queries(path=args.queries)
candidate_document_id_set = [x for sublist in runs.values() for x in sublist]
if os.path.isdir(args.corpus):
    collections = load_collections(dir=args.corpus)
else: 
    #path
    collections = load_collections(path=args.corpus)


# n_passage = 0
with open(args.output_text_pair, 'w') as text_pair, open(args.output_id_pair, 'w') as id_pair:
    for i, (qid, docid_ranklist) in enumerate(runs.items()):

        for k, docid in enumerate(docid_ranklist):
            if args.doc_level:
                pass
            else:
                text_pair.write(
                        f"Query: {queries[qid]} Document: {collections[docid]} Relevant:\n"
                )
                id_pair.write(
                        f"{qid}\t{docid}\t{k+1}\n"
                )

            # n_passage += 1
        
        if i % 1000 == 0:
            print(f'Creating re-ranking input ...{i}')
            # print('Creating T5-qp-ranking-pairs...{}'.format(n_passage))

print("DONE!")
