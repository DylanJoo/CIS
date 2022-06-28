import json
import argparse
import collections
import os

def main(args):

    canard = json.load(open(args.path_canard, 'r'))
    outputs = {'Rewrite': open(args.path_canard.replace('json','rewrite.tsv'), 'w'),
               'Question': open(args.path_canard.replace('json', 'utterance.tsv'), 'w'),
               'Answer': open(args.path_canard.replace('json', 'answer.tsv'), 'w')}

    for i, dict_canard in enumerate(canard):

        quac_id = f"{dict_canard['QuAC_dialog_id']}_q#{dict_canard['Question_no']-1}"
        assert QUAC_ANS[quac_id]['Question'] == dict_canard['Question'], 'Mismatched'
        for col in args.column_2:
            if col == 'Answer':
                resource = f"{dict_canard['Rewrite']} {QUAC_ANS[quac_id][col]}"
            else:
                resource = dict_canard[col]
            outputs[col].write(f"{quac_id}\t{resource}\n")

        if i % 10000 == 0:
            print("{} finished...".format(i))

def parse_quac():
    quac = json.load(open('quac/train_quac.json', 'r'))['data']+json.load(open('quac/val_quac.json', 'r'))['data'] 
    data = collections.defaultdict(dict)
    for topic in quac:
        i = 0
        for turn in topic['paragraphs'][0]['qas']:
            turn_id = turn['id']
            if turn_id.split("q#")[1] != str(i):
                print("[FIX] Incorrent turn numbers found, fix QuAC turn number")
                turn_id = f'{turn_id.split("q#")[0]}q#{i}'

            if turn['id'] == "C_2ca59977d66d4742939232f443ceda41_1_q#6":
                print(f"[FIX] Ambiguous question: {turn['question']}, ignore this turn.")
                i -= 1
            else:
                data[turn_id] = {
                        'Answer': turn['orig_answer']['text'].replace("CANNOTANSWER", ""), 
                        'Question': turn['question']
                }
            i += 1
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-canard", "--path_canard", default="canard/train.json", type=str)
    parser.add_argument("-col2", "--column_2", action='append', default=['Question', 'Rewrite', 'Answer'])
    args = parser.parse_args()

    QUAC_ANS = parse_quac() if 'Answer' in args.column_2 else None

    main(args)
