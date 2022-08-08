import json
import argparse
import collections
import os

def main(args):

    canard = json.load(open(args.path_canard, 'r'))
    outputs = {}
    for col in args.column_2:
        outputs.update({
                col: open(args.path_canard.replace('json', f'{col.lower()}.tsv'), 'w')
        })

    for i, dict_canard in enumerate(canard):

        quac_id = f"{dict_canard['QuAC_dialog_id']}_q#{dict_canard['Question_no']-1}"
        assert QUAC_ANS[quac_id]['Question'] == dict_canard['Question'], 'Mismatched'
        for col in set(args.column_2):
            if col == 'Answer': # beside answer, append the rewrite question before
                resource = f"{dict_canard['Rewrite']} {QUAC_ANS[quac_id][col]}"
            elif col == 'History': # include topic context and response context
                if args.full_context: 
                    context = dict_canard[col][2:]
                else: # only question context (i.e. index 2, 4, 6, ...)
                    context = [c for i, c in enumerate(dict_canard[col][2:]) if i % 2 == 0]
                resource = "|".join(context) + f"[Q] {dict_canard['Question']}"
            else:
                resource = dict_canard[col]
            outputs[col].write(f"{quac_id}\t{resource}\n")

        if i % 10000 == 0:
            print("{} finished...".format(i))

def parse_quac(dir):
    quac = json.load(open(os.path.join(dir, 'train_quac.json'), 'r'))['data'] + \
            json.load(open(os.path.join(dir, 'val_quac.json'), 'r'))['data'] 
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
    parser.add_argument("-canard", "--path_canard", default="data/canard/train.json", type=str)
    parser.add_argument("-quac", "--dir_quac", default="data/quac/", type=str)
    parser.add_argument("-col2", "--column_2", action='append', default=['Question'])
    parser.add_argument("-full", "--full_context", action='store_true', default=False)
    args = parser.parse_args()

    QUAC_ANS = parse_quac(args.dir_quac)

    main(args)
