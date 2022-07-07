# make directory
mkdir cqe/checkpoints 

# download checkpoints
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz -P cqe/checkpoints/
tar -xvzf cqe/checkpoints/colbertv2.0.tar.gz 

# clean it up
rm cqe/checkpoints/colbertv2.0.tar.gz 

# download dataset
gsutil cp gs://cfdaclip-tmp/cast/canard_convir.train.triples.cqe/* convir_data/
gsutil cp gs://cfdaclip-tmp/cast/canard_convir.train.quadruples.cqe/* convir_data/
