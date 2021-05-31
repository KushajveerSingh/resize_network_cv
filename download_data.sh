mkdir -p data

# Imagenette install instructions
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -xzf imagenette2.tgz -C data/
rm imagenette2.tgz

# Imagewoof install instructions
wget https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz
tar -xzf imagewoof2.tgz -C data/
rm imagewoof2.tgz