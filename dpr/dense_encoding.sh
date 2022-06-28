export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

python3 -m pyserini.encode \
  input   --corpus ../data/trec-car+marco-psg \
          --field text \
          --delimiter "no-to-split" \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings ./embeddings \
  encoder --encoder castorini/tct_colbert-v2-hnp-msmarco \
          --fields text \
          --batch 128 \
          --fp16

