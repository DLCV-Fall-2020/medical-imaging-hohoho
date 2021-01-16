CH=3
BSIZE=48
DEVICE=cuda 

train_simclr_resnet18: 
	python3 train.py --backbone resnet18 --bsize $(BSIZE) --ch $(CH) --device $(DEVICE) --fp16_precision 

