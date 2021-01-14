
######################## Training ############################

BACKBONE=resnetxx_3d
THRESHOLD=0.5
DEVICE=cuda 
BSIZE=4
MODEL_PATH=./checkpoints/resnet18/best.pth

train_resnet18_lstm: 
	$(eval BACKBONE=resnet18)
	python3 train.py --backbone $(BACKBONE) --bsize $(BSIZE) --device $(DEVICE) --fp16_precision
	$(MAKE) submit BACKBONE=resnet18 

train_resnet34_lstm:
	$(eval BACKBONE=resnet34)
	python3 train.py --backbone $(BACKBONE) --bsize $(BSIZE) --device cuda --fp16_precision 
	$(MAKE) submit BACKBONE=resnet34 

train_resnet50_lstm:
	$(eval BACKBONE=resnet50)
	python3 train.py --backbone $(BACKBONE) --bsize 16 --device cuda --fp16_precision 

######################## Inference ############################

CUDA_ENV=export CUDA_VISIBLE_DEVICES=0

submit:
	$(CUDA_ENV) && python3 inference.py --backbone $(BACKBONE) --model_path $(MODEL_PATH) --pred_csv_path ./pred.csv --threshold $(THRESHOLD) --device $(DEVICE)
	python3 to_kaggle.py --pred_csv_path ./pred.csv --out_csv_path ./submit.csv 
	rm ./pred.csv 

test:
	python3 test.py 

