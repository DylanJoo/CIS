import os
import collections
import re
import argparse
import json
from utils import load_runs, load_topics, load_collections

def convert_run_to_conv_monot5(args):
    # laod runs
    runs = load_runs(args.run)

    # load evlauation topics
    topics = load_topics(args.topic_queries)

    if os.path.isdir(args.corpus):
        collections = load_collections(dir=args.corpus)
    else: 
        collections = load_collections(path=args.corpus)

    # n_passage = 0
    with open(args.output_jsonl, 'w') as f:
        for i, (qid, docid_ranklist) in enumerate(runs.items()):

            topics[qid].pop('canonical_passage_id')
            topics[qid].pop('history_utterances')
            topics[qid].pop('history_responses')
            topics[qid].update({"context": "|".join(topics[qid]['context'])})

            for k, docid in enumerate(docid_ranklist):

                topics[qid].update({"passage": collections[docid]})
                f.write(json.dumps(topics[qid])+'\n')
            
            if i % 1000 == 0:
                break
                print(f'Creating re-ranking input ...{i}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-run", "--run", type=str,)
    parser.add_argument("-corpus", "--corpus", type=str,)
    parser.add_argument("-topic", "--topic_queries", type=str,)
    parser.add_argument("-output", "--output_jsonl", type=str,)
    args = parser.parse_args()

    convert_run_to_conv_monot5(args)
    print("DONE!")
