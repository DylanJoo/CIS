import collections
import json
from spacy.lang.en import English
import os
import re
# import tensorflow.compat.v1 as tf

nlp = English()


def combine_utterance_response(utterances, responses_p, responses_a=None, current_i=-100):
    '''Indicate the i-th turn would consist i-1, i-2, i-3'''
    if responses_a is None:
        responses_a = [None] * len(utterances)
    output = list()
    for i, (u, rp, ra) in enumerate(zip(utterances[:-1], responses_p[:-1], responses_a[:-1])):
        if i >= (current_i - 3):
            output.append("{} ||| {}".format(u, rp))
        elif ra:
            output.append("{} ||| {}".format(u, ra))
        else:
            output.append("{}".format(u))
    
    output.append("{}".format(utterances[-1]))
    output = " ||| ".join(output)
    output = " ".join([tok.text for tok in nlp(output)])
    return output

def merge_utterance(utterances):
    '''Only use the raw utterances.'''
    utterances = " ||| ".join(utterances)
    utterances = " ".join([tok.text for tok in nlp(utterances)])

    return utterances

def get_eval_corpus(corpus_path, eval_data1_path, eval_data2_path, filename):
    corpus_dict = collections.defaultdict(str)
    corpus_candidates = set()
    
    # Get the eval 2020 corpus (in canonical)
    eval_data = json.load(open(eval_data1_path, 'r'))
    for topic in eval_data:
        for turn in topic['turn']:
            corpus_dict[str(turn['manual_canonical_result_id'])] = ""
            corpus_candidates.add(str(turn['manual_canonical_result_id']))

    eval_data = json.load(open(eval_data2_path, 'r'))
    for topic in eval_data:
        for turn in topic['turn']:
            corpus_dict[str(turn['automatic_canonical_result_id'])] = ""
            corpus_candidates.add(str(turn['automatic_canonical_result_id']))

    # Traverse all line in the corpus to fetch the canonical
    with open(corpus_path, 'r') as corpus:
        for i, line in enumerate(corpus):
            try:
                pid, ptext = line.strip().split("\t", 1)
            except:
                pid, ptext = None, ""

            if str(pid) in corpus_candidates:
                corpus_dict[str(pid)] = ptext.strip()
                corpus_candidates.remove(str(pid))
            # If all canonical passages are found, then stop traverse
            if len(corpus_candidates) == 0:
                break
            if (i % 1000000) == 1:
                print(f"{i} had been traversed... {len(corpus_candidates)} left.")


    # Save to dictionary
    print(len(corpus_candidates))
    print(corpus_candidates)
    with open(filename, 'w') as f:
        json.dump(corpus_dict, f)

def convert_trecwab_to_t5ntr(json_path, automatic_json_path, collection_path,
                             urels_path, utterance_path,
                             history_path_auto, history_path_auto_cano, 
                             queries_path_auto_cano, queries_path_manu, 
                             answer_cano_json_path):

    data = json.load(open(json_path, 'r'))
    collections = json.load(open(collection_path, 'r'))
    if automatic_json_path:
        data2 = json.load(open(automatic_json_path, 'r'))

    try:
        answers = json.load(open(answer_cano_json_path, 'r'))
        answers_flag = True
        print("Using the processed canonical answer")
        # Load the passage corpus.
    except:
        answers_flag = False
        print("Not using the processed canonical answer, using passage")

    with open(utterance_path, 'w') as u_file, \
    open(history_path_auto, 'w') as ha_file, \
    open(history_path_auto_cano, 'w' ) as hac_file, \
    open(queries_path_auto_cano, 'w') as qac_file, \
    open(queries_path_manu, 'w') as qm_file, \
    open(urels_path, 'w') as urels_file ,\
    open(urels_path+"_manual", 'w') as urels_doc_file:
        # <topic> ||| <subtopic> ||| History utterance ||| utterance
        # TopicID-TurnID \t Rewritten query
        # TopicID-TurnID \t Raw utterance
        for topic_idx, topic in enumerate(data):
            topic_id = topic['number']
            # topic_id = data[topic_idx]['number']
            # history = "<topic> ||| <subtopic>" 
            history = ""
            passages_cano = list()
            answers_cano = list()
            utterances = list()

            for turn_idx, turn in enumerate(topic['turn']):
                turn_id = turn['number']
                if turn_id != turn_idx + 1 : # The wrong query index
                    print("Query id correction: {}-{} to {}-{}".format(topic_id, turn_id, topic_id, turn_idx+1))
                    turn_id = turn_idx + 1
                # id
                topic_turn_id = "{}_{}".format(topic_id, turn_id)
                
                # doucment-passage id (in 2020, passage_id only)
                # document_passage_id = "{}-{}".format(turn['manual_canonical_result_id'], turn['passage_id'])
                if automatic_json_path:
                    passage_id = data2[topic_idx]['turn'][turn_idx]['automatic_canonical_result_id']
                else:
                    passage_id = "{}".format(turn['manual_canonical_result_id'])
                urels_file.write("{} 0 {} 1\n".format(topic_turn_id, passage_id))
                urels_doc_file.write("{} 0 {} 1\n".format(topic_turn_id, turn['manual_canonical_result_id']))

                # rewritten using canonical
                # ground truth
                utterance = turn['raw_utterance'].strip()
                rewritten = turn['automatic_rewritten_utterance'].strip()  
                rewritten_gt = turn['manual_rewritten_utterance'].strip()
                passage_cano = collections[passage_id]

                if answers_flag: 
                    answer_cano = answers[topic_turn_id]
                else:
                    answer_cano = None
                qac_file.write(rewritten + '\n')
                qm_file.write(rewritten_gt + '\n')
                u_file.write("{}\t{}\n".format(topic_turn_id, utterance))
                                
                # canonical passage
                # utterance 
                passages_cano.append(passage_cano)
                answers_cano.append(answer_cano)
                utterances.append(utterance)

                if turn_idx == 0:
                    ha_example = utterance
                    hac_example = utterance
                else:
                    # Only answer
                    ha_example = merge_utterance(utterances)
               
                    # Lag 3 passage
                    hac_example = combine_utterance_response(
                        utterances = utterances, 
                        responses_p = passages_cano, 
                        current_i = turn_idx)
                    
                    # # All answers
                    # hac_example = combine_utterance_response(
                    #     utterances = utterances, 
                    #     responses_p = answers_cano)

                    # Lag 3 passage and answer for the other 
                    # hac_example = combine_utterance_response(
                    #     utterances = utterances, 
                    #     responses_p = passages_cano, 
                    #     responses_a = answers_cano, 
                    #     current_i = turn_idx)
                
                ha_file.write(ha_example + "\n")
                hac_file.write(hac_example + "\n")

if os.path.exists("./data/2020_eval_collections.json"):
    print("Using the fetched evaluation corpus.")
else:
    get_eval_corpus(corpus_path="/tmp2/cychang/trec_2020/CAsT/Conversational-IR/corpus/CAsT_collection/CAsT_collection_2.tsv", 
                    eval_data1_path="./data/2020_manual_evaluation_topics_v1.0.json", 
                    eval_data2_path="./data/2020_automatic_evaluation_topics_v1.0.json", 
                    filename="./data/2020_eval_collections.json")

convert_trecwab_to_t5ntr(
    json_path = "./data/2020_manual_evaluation_topics_v1.0.json",
    automatic_json_path = "./data/2020_automatic_evaluation_topics_v1.0.json",
    collection_path = "./data/2020_eval_collections.json",
    utterance_path = "./data/utterances.tsv",
    history_path_auto = "./data/history.txt",
    history_path_auto_cano = "./data/history_autocano.txt",
    queries_path_auto_cano = "./data/query/queries_autocano.txt",
    queries_path_manu = "./data/query/queries_manual.txt",
    urels_path = "./data/urels.dev.trec",
    answer_cano_json_path = None
)
