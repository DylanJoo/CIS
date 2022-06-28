export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/trec-car+marco-psg \
  --index indexes/doc_content_2020_jsonl \
  --generator DefaultLuceneDocumentGenerator \
  --threads 9
