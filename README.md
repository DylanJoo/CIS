# conversational_information_search

In this repositary, we use TREC CAsT 2020 data for experiments.
source: https://github.com/daltonj/treccastweb/tree/master/2020/

# Data Preparation
Donwload public corpus

Parsing tsv file into jsonl
```
python3 tools/convert_msmarco_psg_to_jsonl.py \
  --collection-path /tmp2/jhju/datasets/msmarco-psgs/collection.tsv \
   -l-output-folder data/cast22/collections/
```
