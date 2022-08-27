import os
import argparse
import random
import collections
import json
from utils import load_queries, load_runs, load_collections, doc_pool_random_sampling, load_topics, normalized

parser = argparse.ArgumentParser()
parser.add_argument("--topic", default="data/canard/train.queries.jsonl", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()


fout = open(args.output, 'w')

with open(args.topic) as topic:
    for i, line in enumerate(topic):
        query_dict = json.loads(line.strip())
        qid = query_dict['id']
        q = normalized(query_dict['utterance'])
        c_t = "|".join(query_dict['history_topic'])
        c_u = query_dict['history_utterances']
        c_r = query_dict['history_responses']
        c = normalized("|".join([c_t] + [f"{u}|{r}" for u, r in zip(c_u, c_r)]))

        rw = normalized(query_dict['rewrite'])

        fout.write(f"Query: {q} Context: {c} Rewrite:\t{rw}\n")

    if i % 1000 == 0:
        print(f"{i} canard ntr examples finished...")

fout.close()

print("DONE")

