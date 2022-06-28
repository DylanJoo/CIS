export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# for query_type in utterance rewrite answer;do
for query_type in answer rewrite;do
  python3 sparse_retrieval.py \
      -k 1000 -k1 0.82 -b 0.68 \
      -index indexes/treccar_marcopsg \
      -query ../data/canard/train.${query_type}.tsv \
      -output runs/train.canard.${query_type}.top1000.trec
done
