# make directory
mkdir colbert/checkpoints 
mkdir cqe/checkpoints 

# download checkpoints
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz -P colbert/checkpoints/
tar -xvzf colbert/checkpoints/colbertv2.0.tar.gz 

# clean it up
rm colbert/checkpoints/colbertv2.0.tar.gz 
cd cqe/checkpoints
ln -s colbert/checkpoints/colbertv2.0

# download dataset
gsutil cp gs://cfdaclip-tmp/cast/canard_convir.train.triples.cqe/* convir_data/
gsutil cp gs://cfdaclip-tmp/cast/canard_convir.train.quadruples.cqe/* convir_data/
