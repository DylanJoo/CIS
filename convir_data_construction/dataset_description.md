Record the variants of transformed CANARD dataset for IR

* canard_convir.train.convtriples.cqe.v0.jsonl
* canard_convir.train.convtriples.cqe.v1.jsonl
* canard_convir.train.triples.cqe.v0.jsonl
* canard_convir.train.triples.cqe.v1.jsonl
* sample.jsonl
* train.triples.sample.jsonl

- Input variants:
    - For CQE (triples): 
    ```
    {'query': qtext, 'pos_passage': pos_ptext, 'neg_passage': neg_ptext}
    *qtext: utterance_1|utterance_2| ....|[Q] utterance_t
    ```
    - For LICQE (quadruples): 
    ```
    {'utterance': utext, 'context': ctext, 'pos_passage': pos_ptext, 'neg_passage': neg_ptext}
    *ctext: utterance_1|utterance_2| ....|utterance_{t-1}
    *utext: utterance_t
    ```
- Version variants:
    - original: 
        (1) CANARD written queries 
        (2) BM25 Top1000 passages(CAsT 2020) 
        (3) ColBert reranking
        (4) Query by history raw utterances (no response)
        (4) Positive pool from top3 
        (5) Negative pool from the remaining 197
        (6) 100 Synthetic convir triples per query
    - v0: 
        (1) CANARD written queries 
        (2) BM25 Top1000 passages(CAsT 2020) 
        (3) monoT5 reranking
        (4) Query by history raw utterances (no response)
        (4) Positive pool from top3 
        (5) Negative pool from the remaining 197
        (6) 100 Synthetic convir triples per query
    - v1: 
        (1) CANARD written queries 
        (2) BM25 Top1000 passages(CAsT 2020) 
        (3) monoT5 reranking
        (4) Query by history raw utterances (no response)
        (4) Positive pool from overlap of (answer+rewrite && rewrite)
        (5) Negative pool from diff set of 
        (6) 100 Synthetic convir triples per query
