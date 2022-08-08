import json
import argparse
import collections
import os
from spacy.lang.en import English
import spacy

def combine_context(history, topics=None, context_window=0):
    if context_window is not None:
        output = history[-context_window * 2:]
    if topics is not None:
        output = topics + output

    return " ||| ".join(output)

def tokens_expansion(rewrite_query, expansion_source, diff=True):
    if diff:
        output = " ".join([token for token in expansion_source \
        if token not in raw_query.lower()])

    pos = spacy.load("en_core_web_sm")
    output = pos(expansion_source.lower())
    entities = [token.text for token in output \
            if (token.pos_ in ['NOUN', 'PROPN'])]
    return " | ".join(entities)

def make_canard_dataset(args):

    canard = json.load(open(args.path_canard, 'r'))
    output = open(args.path_output, 'w')

    for i, dict_canard in enumerate(canard):
        # context: 
        topic_context = dict_canard['History'][:2]
        utterance_context = dict_canard['History'][2:]
        ## [TODO] response context: [Optional] system repsonse from open domain retrieval

        utterance = dict_canard['Question']
        rewrite = dict_canard['Rewrite']
        quac_id = dict_canard['QuAC_dialog_id']
        turn_id = dict_canard['Question_no']
        quac_turn_id = "{}_q#{}".format(quac_id, turn_id)

        ## Prepare input suequence and output target
        ## (1) [Baseline] topic context + question/response context + question --> rewrite
        src = combine_context(utterance_context, topic_context, 0)
        tgt = rewrite

        ## (2) [Utterance only] topic context + question context + question --> rewrite
        if args.question_only:
            utterance_question_context = [u for i, u in enumerate(utterance_context) if i % 2 == 0]
            src = combine_context(utterance_question_context, topic_context, 0)

        ## (3) [Repsponse only] topic context + question context + question --> rewrite
        ### Deprecated bc logically wrong

        ## (4) [Token generation] (1) setting + token generation 
        if args.query_expansion == 'answer':
            tgt = rewrite + " ||| " + tokens_expansion(rewrite, answer)
        if args.query_expansion == 'history':
            tgt = rewrite + " ||| " + tokens_expansion(rewrite, src)

        output.write(f"{src}\t{tgt}\n")

        if i % 1000 == 0:
            print("{} finished...".format(i))


parser = argparse.ArgumentParser()
parser.add_argument("-canard", "--path_canard", default="train_canard.json", type=str)
parser.add_argument("-conv_qa", "--path_conv_qa", default="train_convqa.json", type=str)
parser.add_argument("-out", "--path_output", default="train_canard+.json", type=str)
parser.add_argument("--spacy", action="store_true", default=False)
parser.add_argument("--question_only", action="store_true", default=False)
parser.add_argument("--reverse", action="store_true", default=False)
parser.add_argument("-response", action="store_true", default=False)
parser.add_argument("-qe", "--query_expansion", type=str, default=None)
args = parser.parse_args()

print(args)
make_canard_dataset(args)
print("DONE")

