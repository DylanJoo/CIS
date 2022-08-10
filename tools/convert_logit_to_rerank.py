import collections
import tensorflow.compat.v1 as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-flogits", "--path_false_logit", type=str)
parser.add_argument("-tlogits", "--path_true_logit", type=str)
parser.add_argument("-score", "--path_score", type=str)
parser.add_argument("-runs", "--path_runs", type=str)
parser.add_argument("-topk", default=1000, type=int)
parser.add_argument("-rerank_runs", "--path_rerank_runs", type=str)
parser.add_argument("--resoftmax", action="store_true", default=True)
args = parser.parse_args() 

def convert_logit_to_prob(args):

    with tf.io.gfile.GFile(args.path_score, 'w') as f, \
    tf.io.gfile.GFile(args.path_true_logit, "r") as true_logits, \
    tf.io.gfile.GFile(args.path_false_logit, "r") as false_logits:

        for i, (true_logit, false_logit) in enumerate(zip(true_logits, false_logits)):
            true_prob = np.exp(float(true_logit))
            false_prob = np.exp(float(false_logit))
            sum = true_prob + false_prob
            
            if args.resoftmax:
                true_prob = true_prob / sum
                false_prob = false_prob / sum

            f.write("{:.16f}\t{:.16f}\t{:.16f}\n".format(
                true_prob, false_prob, np.add(true_prob, false_prob)))

            if i % 1000000 == 0:
                print("[Re-ranker] {} query-passage pair had been scored.".format(i))

def rerank_runs(args):

    query_candidates = collections.defaultdict(list) 
    with tf.io.gfile.GFile(args.path_score, 'r') as score_file, \
    tf.io.gfile.GFile(args.path_runs, "r") as baseline_run_file:

        for i, (score_line, run_line) in enumerate(zip(score_file, baseline_run_file)):
            true_prob, false_prob, _ = score_line.rstrip().split('\t')
            qid, _, docid, order, _, _ = run_line.rstrip().split()
            if int(order) <= args.topk:
                query_candidates[qid].append((docid, true_prob, false_prob))
    '''example: query_candidate[query7777] = [(doc1111, 0.98, 0.02), (doc2222, 0.99, 0.01), ....]
    '''

    with tf.io.gfile.GFile(args.path_rerank_runs, 'w') as f:
        for i, (qid, candidate_passage_list) in enumerate(query_candidates.items()):
            # Using true prob as score, so reverse the order.
            candidate_passage_list = sorted(candidate_passage_list, key=lambda x: x[1], reverse=True)
            for idx, (docid, true_prob, false_prob) in enumerate(candidate_passage_list[:1000]):
                example = '{} Q0 {} {} {} monot5\n'.format(qid, docid, str(idx+1), true_prob)

                f.write(example)

            if i % 100 == 0:
                print('[Re-ranker] Ranking passages...{}'.format(i))

if tf.io.gfile.exists(args.path_runs) is False:
  print("Invalid path of run file")
  exit(0)

convert_logit_to_prob(args)
print("Score finished")
rerank_runs(args)
print("DONE")
