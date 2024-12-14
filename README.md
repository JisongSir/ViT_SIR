# ViT-SIR
Vision Transformer-based Shoe Image Retrieval with Enhanced Feature Representation
<img src="ViT_SIR.png" alt="ViT_SIR" style="width: 70%;"/>
# Installation
pip install -r requirements.txt
# Train
python train.py --config_file configs 'XXX' \
eg.python train.py --config_file configs/Shoes/vit_train.yml
# Test
python test.py --config_file 'XXX' TEST.WEIGHT 'XXX'\
eg.python test.py --config_file configs/Shoes/vit_test.yml TEST.WEIGHT 'choose the trained weight'
# SIR-PPSUC Dataset
The SIR-PPSUC dataset used in this project can be accessed by submitting a real-name request to the project leader via email: tangyunqi@ppsuc.edu.cn.
