export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

python3 spr/sparse_retrieval.py \
  -k 1000 -k1 0.82 -b 0.68 \
  -index indexes/cast20 \
  -query data/cast20/2020_evaluation_topics_v1.0.jsonl \
  -output runs/cast20.automatic.rewrite.spr.top1000.trec \
  -qval automatic_rewritten &

python3 spr/sparse_retrieval.py \
  -k 1000 -k1 0.82 -b 0.68 \
  -index indexes/cast20 \
  -query data/cast20/2020_evaluation_topics_v1.0.jsonl \
  -output runs/cast20.manual.rewrite.spr.top1000.trec \
  -qval manual_rewritten
