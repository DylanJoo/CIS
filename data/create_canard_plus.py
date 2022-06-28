import json
import argparse
import collections
import os
from spacy.lang.en import English
import spacy

parser = argparse.ArgumentParser()
parser.add_argument("-canard", "--path_canard", default="train_canard.json", type=str)
parser.add_argument("-quac-tr", "--path_quac_train", default="train_quac.json", type=str)
parser.add_argument("-quac-va", "--path_quac_val", default="val_quac.json", type=str)
parser.add_argument("-conv_qa", "--path_conv_qa", default="train_convqa.json", type=str)
parser.add_argument("-out", "--path_output", default="train_canard+.json", type=str)
parser.add_argument("--spacy", action="store_true", default=False)
parser.add_argument("--question_only", action="store_true", default=False)
parser.add_argument("--reverse", action="store_true", default=False)
parser.add_argument("-rewrite", action="store_true", default=False)
parser.add_argument("-response", action="store_true", default=False)
parser.add_argument("-eexp", "--entity_expansion", action="store_true", default=False)
parser.add_argument("-eext", "--entity_extraction", action="store_true", default=False)
args = parser.parse_args()


def convert_quac_to_conv_qa(args):
    data = open(args.path_quac_train, 'r')
    quac = json.load(data)['data']
    data = open(args.path_quac_val, 'r')
    quac = quac + json.load(data)['data']

    conversational_qa = open(args.path_conv_qa, 'w')
    conv_qa_dict = collections.defaultdict()

    for i_topic, topic in enumerate(quac):
        # Topic related data
        background = topic['background']
        title = topic['title']
        section_title = topic['section_title']

        # Turn related data
        content = topic['paragraphs'][0]
        context = content['context']
        for i_turn, turn in enumerate(content['qas']):
            question = turn['question']
            # The "natrual language-like answer'
            answer = turn['answers'][0]['text'].replace("CANNOTANSWER", "I don't know.")
            # THe "Original" answert from the given context
            orig_answer = turn['orig_answer']['text'].replace("CANNOTANSWER", "I don't know.")
            
            question_id = turn['id']
            conversation_id, turn_id = question_id.split("_q#")
            # Some turn index is wrong in QuAC
            if int(turn_id) != i_turn:
                print("Mismatch found in {}, Corrected from q#{} to q#{}".format(conversation_id, turn_id, i_turn))
                question_id = "{}_q#{}".format(conversation_id, i_turn) 
            conv_qa_dict[question_id] = {"context": context, "question": question, 
                    "answer": answer,"orig_answer": orig_answer}
    
    json.dump(conv_qa_dict, conversational_qa) 
    print("{} have been converted...".format(args.path_conv_qa))

# History 1: Using the lag "answer " passage for exapnding context
def combine_utterance_response(utterances, responses, pre_history, current_i=-100):
    '''Indicate the i-th turn would consist i-1, i-2, i-3'''
    output = list()
    for i, (u, r) in enumerate(zip(utterances[:-1], responses[:-1])): 
        if i >= (current_i - 1):
            output.append(u)
            output.append(r)
            # output.append("{} : {}".format(u, r))
        else:
            output.append(u)
    # If we need the pre-history like title or descprition.
    if len(pre_history):
        output = pre_history + output

    output.append(utterances[-1])
    output = " ||| ".join(output)

    return output

# History 2: Using the lag "entities" of lag turn's answer for expanding context
# def combine_utterance_entity(utterances, responses, pre_history, current_i=-100):
#     '''Indicate the i-th turn would consist i-1, i-2, i-3, 
#     and also using spacy to extract the entites as response'''
#
#     output = list()
#     for i, (u, r) in enumerate(zip(utterances[:-1], responses[:-1])):
#         if i >= (current_i - 3):
#             # output.append(u)
#             # entities = [token.text for token in nlp(r)]
#             entities = [token for token in r.split() if token.lower() not in u.lower()]
#             # print(entities)
#             output.append(u + " ||| " + " ".join(entities))
#         else:
#             output.append(u)
#     
#     # If we need the pre-history like title or descprition.
#     if len(pre_history):
#         output = pre_history + output
#
#     output.append(utterances[-1])
#     output = " ||| ".join(output)
#     
#     return output

# Rewrite 1
def sentence_concatenate(rewrite_query, concat_source): 
    '''Appending the targeting sentences.
    '''
    return "{} ||| {}".format(rewrite_query, concat_source)

# Rewrite 2
def omission_tokens(raw_query, token_source, diff=False): 
    '''Extract the additional token with information.
    '''
    if diff:
        output = " ".join([token for token in token_source \
        if token not in raw_query.lower().split()])

    output = pos(token_source.lower())
    omission = [token.text for token in output \
            if (token.pos_ in ['NOUN', 'PROPN'])]
    return " | ".join(omission)

# Rewrite 3
def tokens_expansion(rewrite_query, expansion_source):
    '''Extract the additional token with information.
    '''
    output = pos(expansion_source.lower())
    entities = [token.text for token in output \
            if (token.pos_ in ['NOUN', 'PROPN'])]
    return "{} ||| {}".format(rewrite_query, " | ".join(entities))

def merge(args):

    conv_qa = json.load(open(args.path_conv_qa, 'r'))
    canard = json.load(open(args.path_canard, 'r'))
    output = open(args.path_output, 'w')
    if args.reverse:
        output_reverse = open(args.path_output + "-reverse", 'w')
    quac_id = ""

    for i, dict_canard in enumerate(canard):
        if dict_canard['QuAC_dialog_id'] != quac_id:
            new_topic = True
        # Although the history is already containt the Q and A
        history = dict_canard['History'][:2]
        question = dict_canard['Question']
        rewrite = dict_canard['Rewrite']
        quac_id = dict_canard['QuAC_dialog_id']
        turn_id = int(dict_canard['Question_no']) - 1
        quac_turn_id = "{}_q#{}".format(quac_id, turn_id)
        
        # If new topic, clean the answers list, and new another answer list
        if new_topic:
            answers = list()
            questions = list()
            rewrites = list()
            new_topic = False

        qa = conv_qa[quac_turn_id]
        context = qa['context']
        questions += [question]
        answer = qa['answer'] 
        answers += [answer]
        rewrites += [rewrite] 

        # coreference resolution
        src_coref = combine_utterance_response(questions, answers, history)
        tgt_coref = rewrite

        if args.question_only: # no responses would be append
            src_coref = combine_utterance_response(questions, answers, history, current_i=len(questions))
        
        if args.reverse:
            src_coref_re = combine_utterance_response(rewrites, answers, history)
            tgt_coref_re = question
            output_reverse.write("{}\t{}\n".format(src_coref_re, tgt_coref_re))

        if args.response:
            tgt_coref = sentence_concatenate(rewrite, answer)
            if args.entity_expansion:
                tgt_coref = tokens_expansion(rewrite, answer)
            if args.entity_extraction:
                tgt_coref = omission_tokens(rewrite, answer)
        
        if args.rewrite:
            #tgt_coref = sentence_concatenate(rewrite, rewrite)
            if args.entity_expansion:
                tgt_coref = tokens_expansion(rewrite, rewrite)  
            if args.entity_extraction:
                tgt_coref = omission_tokens(rewrite, rewrite)

        if args.spacy:
            src_coref = ' '.join([tok.text for tok in nlp(src_coref)])
            tgt_coref = ' '.join([tok.text for tok in nlp(tgt_coref)])

        output.write("{}\t{}\n".format(src_coref, tgt_coref))

        if i % 1000 == 0:
            print("{} finished...".format(i))
        # question answering
        #example_qa = "Response: {} Query: {} Rewrite:\n".format()

print(args)
if args.response and args.rewrite:
    print("Cannot use both expansion source.")
    exit(0)
if args.entity_extraction and args.entity_expansion:
    print("Cannot use both expansion strategy.")
    exit(0)

if args.entity_extraction or args.entity_expansion:
    pos = spacy.load("en_core_web_sm")
if args.spacy:
    nlp = English()
#if os.path.isfile(args.path_conv_qa) is False:
convert_quac_to_conv_qa(args)
merge(args)
print("DONE")

