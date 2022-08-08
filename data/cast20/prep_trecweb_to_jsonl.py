import collections
import json
import os
import re

PATH_AUTOMATIC='2020_automatic_evaluation_topics_v1.0.json'
PATH_MANUAL='2020_manual_evaluation_topics_v1.0.json'
PATH_EVAL_COLLECTION='2020_eval_collections.json'
PATH_OUTPUT='2020_evaluation_topics_v1.0.jsonl'

def convert_trecweb_to_jsonl():
    manual = json.load(open(PATH_MANUAL, 'r'))
    automatic = json.load(open(PATH_AUTOMATIC, 'r'))
    passage_collections = json.load(open(PATH_EVAL_COLLECTION, 'r'))

    output = open(PATH_OUTPUT, 'w')

    history = {"context": [], "utterances": [], "responses": []}

    for topic_i, topic in enumerate(manual):
        topic_id = topic['number']

        for turn_i, turn in enumerate(topic['turn']):
            turn_id = turn['number']

            if (turn_i+1) != turn['number']:
                print(f"Query id correction: {topic_id}-{turn_id} to {topic_id}-{turn_idx+1}")
                turn_id = turn_idx + 1

            # information
            utterance = turn['raw_utterance'].strip()
            automatic_rewritten = turn['automatic_rewritten_utterance'].strip()
            manual_rewritten = turn['manual_rewritten_utterance'].strip()
            canonical_passage_id = \
                    automatic[topic_i]['turn'][turn_i]['automatic_canonical_result_id']
            passage_cano = passage_collections[canonical_passage_id]
            
            # output
            output.write(
                    json.dumps({'id': f"{topic_id}_{turn_id}",
                                'utterance': utterance,
                                'automatic_rewritten': automatic_rewritten,
                                'manual_rewritten': manual_rewritten,
                                'canonical_passage_id': canonical_passage_id,
                                'context': history['context'],
                                'history_utterances': history['utterances'],
                                'history_responses': history['responses']}) +'\n'
            )

            # history
            history['context'].append(utterance)
            history['context'].append(passage_cano)
            history['utterances'].append(utterance)
            history['responses'].append(passage_cano)

convert_trecweb_to_jsonl()
