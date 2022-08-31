import os
import collections
import argparse
import json
from utils import load_runs, load_collections, load_queries, load_topics, normalized, document_extraction

def convert_to_monot5(args):

    runs = load_runs(args.run)
    topics = load_topics(args.topic_queries)
    collections = load_collections(dir=args.corpus) # note this collection is composed of chuncks
    if args.reformulated_queries:
        queries = load_queries(args.reformulated_queries)

    with open(args.output_text_pair, 'w') as text_pair, \
         open(args.output_id_pair, 'w') as id_pair:

        for i, (qid, pid_ranklist) in enumerate(runs.items()):
            for k, pid in enumerate(pid_ranklist):

                P = normalized(collections[pid])

                # Query source 
                if args.reformulated_queries:
                    Q = normalized(queries[qid])
                else:
                    Q = normalized(topics[qid][f'{args.use_query_key}'])

                ## (1) Conversational query for conv monot5
                if args.use_context is not None:
                    HU = topics[qid]['history_utterances']
                    HR = topics[qid]['history_responses']

                    if args.use_response:
                        # [TODO 1] Make the document shorter to meet the lenght limiation of T5
                        # [TODO 2] Make the all query and k responses
                        C = "|".join([f"{u}|{r}" for u, r in zip(HU, HR)][-args.use_context:])
                    else:
                        C = "|".join([u for u in HU[-args.use_context:]])
                    C = normalized(C)

                    id_pair.write(f"{qid}\t{pid}\n")
                    text_pair.write(f"Query: {Q} Context: {C} Document: {P} Relevant:\n")

                ## (2) Normal standalone query for monot5
                else:
                    id_pair.write(f"{qid}\t{pid}\n")
                    text_pair.write(f"Query: {Q} Document: {P} Relevant:\n")
                    
            if i % 1000 == 0:
                print(f'Creating re-ranking input ...{i}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True,)
    parser.add_argument("--corpus", type=str, required=True,)
    parser.add_argument("--topic_queries", type=str, required=True,)
    parser.add_argument("--reformulated_queries", type=str, required=False,)
    parser.add_argument("--output_text_pair", type=str, required=True,)
    parser.add_argument("--output_id_pair", type=str, required=True,)
    parser.add_argument("--use_context", type=int, default=None)
    parser.add_argument("--use_query_key", type=str, default='utterance')
    parser.add_argument("--use_response", action='store_true', default=False)
    args = parser.parse_args()

    convert_to_monot5(args)
    print("DONE!")
