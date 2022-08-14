import collections
import numpy as np
from tools.utils import load_runs, load_topics
from sklearn.preprocessing import StandardScaler, MinMaxScaler

qstat = collections.defaultdict(list)
runs = load_runs('runs/cast20.automatic.rewrite.spr.top1000.trec', True)
topics = load_topics("data/cast20/2020_evaluation_topics_v1.0.jsonl")

for qid in runs:
    for docid_score in runs[qid]:
        qstat[qid].append(docid_score[1])

with open('stats.txt', 'w') as f:
    for qid in qstat:
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        qstat[qid] = scaler.fit_transform(np.array(qstat[qid]).reshape(-1, 1))
        qstat[qid] = qstat[qid][:1000]
        mean = np.round(np.mean(qstat[qid]), 4)
        std = np.round(np.std(qstat[qid]), 4)
        q1 = np.round(np.quantile(qstat[qid], 0.25), 4)
        q2 = np.round(np.quantile(qstat[qid], 0.5), 4)
        q3 = np.round(np.quantile(qstat[qid], 0.75), 4)

        f.write(f"QID: {qid}\tMEAN: {mean}\tSTD: {std}\tQ1: {q1}\tQ2: {q2}\tQ3:{q3}\t{mean-q2}\n")
        # f.write(f"Utterance: {topics[qid]['utterance']}\t{topics[qid]['automatic_rewritten']}\n")


    
