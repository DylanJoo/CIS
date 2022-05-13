import numpy as np
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("-input", "--path_input_file", default="orconvqa/train.txt")
parser.add_argument("-output", "--path_output_file", default="train.jsonl")
parser.add_argument("-window", "--window_size", type=int, default=None)
parser.add_argument("-negative", "--negative_sampling", default="only_pos")
args = parser.parse_args()


def sort_history(questions_and_responses, window_size,
                 question_prefix="", response_prefix="", 
                 turn_delimiter="", context_delimiter="",
                 use_response=False):

    history = list()

    for turn in questions_and_responses:
        question = turn['question']
        response = turn['answer']['text']
        if use_response:
            history.append(f'{question_prefix}{question.strip()} {response_prefix}{response.strip()}')
        else:
            history.append(f'{question_prefix}{question.strip()}')

    # history = history[None:]
    history = f"{turn_delimiter}".join(history)
    return history + f'{context_delimiter}'

def main(args):

    fout = open(args.path_output_file, 'w')
    fin = open(args.path_input_file, 'r')

    for i_topic, line in enumerate(fin):
        example = json.loads(line.strip())
    
        # arange history questions and responses
        context_history = sort_history(
                questions_and_responses=example['history'],
                window_size=args.window_size,
                question_prefix="[Q]: ", response_prefix=" [A]: ",
                turn_delimiter=" ||| ", context_delimiter="",
                use_response=True
        )

        question_history = sort_history(
                questions_and_responses=example['history'],
                window_size=args.window_size,
                question_prefix="[Q]: ", response_prefix=" [A]: ",
                turn_delimiter=" ||| ", context_delimiter="",
                use_response=False
        )

        # [TODO] alternative negative sampling approach
        for i, (psg, lbl) in enumerate(zip(example['evidences'], example['retrieval_labels'])):
            example_json = {
                    "history": context_history,
                    "last_history": context_history.rsplit(" ||| ")[-1],
                    "question_history": question_history,
                    "question": example['question'],
                    "rewrite": example['rewrite'],
                    "passage": psg.strip(),
                    "label": lbl
            }

            # (1) Sampled only the relevant passage 
            # (2) Sampled all top-k (k=5) passages

            if args.negative_sampling == "only_pos" and lbl == 1:
                fout.write(json.dumps(example_json) + '\n')
            elif args.negative_sampling == "all":
                fout.write(json.dumps(example_json) + '\n')
            else:
                pass
            
        if i_topic % 1000 == 0:
            print(f'{i_topic+1} topics finished')

main(args)
