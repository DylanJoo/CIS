export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ../data/cast20/collections/ \
  --index ../indexes/cast20_jsonl \
  --generator DefaultLuceneDocumentGenerator \
  --threads 9
