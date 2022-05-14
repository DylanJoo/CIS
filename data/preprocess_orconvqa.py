import numpy as np
import json
import argparse
from collections import defaultdict
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("-input", "--path_input_file", default="orconvqa/train.txt")
parser.add_argument("-output", "--path_output_file", default="orconvqa/train.jsonl")
parser.add_argument("-window", "--window_size", type=int, default=None)
parser.add_argument("-sampling", "--sampling", default='only_pos')
parser.add_argument("-n_turns", "--n_turns_aligned", type=int, default=None)
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

def load_orconvqa(args):

    data_dict = defaultdict(list)
    topic_count = defaultdict(int)
    with open(args.path_input_file, 'r') as fin:
        for i_turn, line in enumerate(fin):
            example = json.loads(line.strip())
            topic_id, turn_id = example.pop('qid').split("#")
            # data_dict[(topic_id, turn_id)] = example
            topic_count[topic_id] += 1
            data_dict[(topic_id, topic_count[topic_id])] = example
    return data_dict, topic_count

def main(args):

    test_dict = defaultdict(int)
    fout = open(args.path_output_file, 'w')
    # fin = open(args.path_input_file, 'r')
    data_dict, topic_count = load_orconvqa(args)
    keys = sorted(data_dict.keys(), key=lambda x: x[0] + str(x[1]))

    for k in keys:
        # example = data_dict.pop(k)
        example = data_dict[k]
        topic_id, turn_id = k

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

        # Positive sample and negative samples
        for i, (psg, lbl) in enumerate(zip(example['evidences'], example['retrieval_labels'])):

            if args.sampling == 'all':
                example_json = {
                        "history": context_history,
                        "last_history": context_history.rsplit(" ||| ")[-1],
                        "question_history": question_history,
                        "question": example['question'],
                        "rewrite": example['rewrite'],
                        "passage": psg.strip(),
                        "label": int(lbl)
                }
                fout.write(json.dumps(example_json) + '\n')
                test_dict[topic_id] += 1

            elif args.sampling == 'only_pos' and int(lbl) == 1:
                example_json = {
                        "history": context_history,
                        "last_history": context_history.rsplit(" ||| ")[-1],
                        "question_history": question_history,
                        "question": example['question'],
                        "rewrite": example['rewrite'],
                        "passage": psg.strip(),
                        "label": int(lbl)
                }
                if args.n_turns_aligned is not None and test_dict[topic_id] >= args.n_turns_aligned:
                    pass
                else:
                    fout.write(json.dumps(example_json) + '\n')
                    test_dict[topic_id] += 1

        if args.n_turns_aligned is not None:
            # check if the turn over the pre-defined
            if turn_id <= args.n_turns_aligned:
                if turn_id >= topic_count[topic_id]:
                    example_json.update({
                        'label': -100, # no loss
                        "question": "", "rewrite": "", "passage": "", 
                        "last_history": "", "question_history": "", "history": ""
                    })
                    for _ in range(args.n_turns_aligned - turn_id):
                        fout.write(json.dumps(example_json) + '\n')
                        test_dict[topic_id] += 1

    print(f'Processed {len(topic_count)} topics and {sum(topic_count.values())} in total')
    print(f'Original distribution of turns in topic: {Counter(topic_count.values())}')
    print(f'Processed distribution of turns in topic: {Counter(test_dict.values())}')

main(args)
