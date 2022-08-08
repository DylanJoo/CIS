export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

python3 -m pyserini.index.faiss \
  --input encoded/ \
  --output faiss-indexes/testing
