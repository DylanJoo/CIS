# make directory
mkdir colbert/checkpoints 
mkdir colbert/checkpoints 

# download checkpoints
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz -P colbert/checkpoints/
tar -xvzf colbert/checkpoints/colbertv2.0.tar.gz 

# clean it up
rm colbert/checkpoints/colbertv2.0.tar.gz 
ln -s colbert/checkpoints/colbertv2.0 cqe/checkpoints/
