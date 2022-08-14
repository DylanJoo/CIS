import os
import collections
import argparse
import json
from utils import load_runs, load_queries, load_collections, load_topics, normalized

def convert_to_monot5(args):
    # laod requirments
    runs = load_runs(args.run)
    # load evlauation topics
    # queries = load_queries(args.queries)
    topics = load_topics(args.topic_queries)

    if os.path.isdir(args.corpus):
        collections = load_collections(dir=args.corpus)
    else: 
        collections = load_collections(path=args.corpus)


    # n_passage = 0
    with open(args.output_text_pair, 'w') as text_pair, open(args.output_id_pair, 'w') as id_pair:
        for i, (qid, docid_ranklist) in enumerate(runs.items()):

            for k, docid in enumerate(docid_ranklist):
                # q = queries[qid].strip()
                d = normalized(collections[docid])
                if args.use_context > 0:
                    q = normalized(topics[qid]['utterance'])
                    c_u = topics[qid]['history_utterances']
                    c_r = topics[qid]['history_responses']
                    c = normalized(
                            "|".join([f"{u}|{r}" for u, r in zip(c_u, c_r)][-args.use_context:])
                    )
                    text_pair.write(f"Query: {q} Context: {c} Document: {d} Relevant:\n")
                else:
                    try:
                        q = normalized(topics[qid]['automatic_rewritten'])
                    except:
                        q = normalized(topics[qid]['rewrite'])
                    text_pair.write(f"Query: {q} Document: {d} Relevant:\n")
                    
                id_pair.write(f"{qid}\t{docid}\n")
            
            if i % 1000 == 0:
                print(f'Creating re-ranking input ...{i}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-run", "--run", type=str, required=False,)
    parser.add_argument("-corpus", "--corpus", type=str, required=True,)
    parser.add_argument("-k", "--top_k", type=int, default=1000,)
    # parser.add_argument("-q", "--queries", type=str, required=True,)
    parser.add_argument("-topic", "--topic_queries", type=str, required=True,)
    parser.add_argument("--output_text_pair", type=str, required=True,)
    parser.add_argument("--output_id_pair", type=str, required=True,)
    parser.add_argument("--use_context", type=int, default=0)
    args = parser.parse_args()

    convert_to_monot5(args)
    print("DONE!")
