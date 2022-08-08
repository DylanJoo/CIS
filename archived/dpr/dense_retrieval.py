from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder

encoder = TctColBertQueryEncoder('castorini/tct_colbert-v2-hnp-msmarco')
# searcher = FaissSearcher.from_prebuilt_index(
#     'msmarco-passage-tct_colbert-hnsw',
#     encoder
# )
searcher = FaissSearcher(
        'faiss-indexes/testing', 
        encoder
)
hits = searcher.search('what is a lobster roll')

for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')
