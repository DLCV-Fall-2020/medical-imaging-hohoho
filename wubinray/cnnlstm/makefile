
######################## Training ############################

CH=3
BACKBONE=resnetxx_3d
THRESHOLD=0.5
DEVICE=cuda 
BSIZE=4
TRAIN_VALID_SPLIT=../base/checkpoints/
LOAD_PRETRAINED=../base/checkpoints/resnet18/best.pth

train_resnet18_lstm_pretrained_cnn: 
	$(eval BACKBONE=resnet18)
	python3 train.py --ch $(CH) --backbone $(BACKBONE) --bsize $(BSIZE) --device $(DEVICE) --train_valid_split_pkl $(TRAIN_VALID_SPLIT) --load_pretrained $(LOAD_PRETRAINED) --fp16_precision --lr 1e-4 --eta_min 5e-6 --threshold 0.5
	$(MAKE) submit BACKBONE=resnet18

train_resnet18_lstm_pretrained_cnn_minyang:
	$(eval BACKBONE=resnet18)
	python3 train.py --ch $(CH) --backbone $(BACKBONE) --bsize $(BSIZE) --device $(DEVICE) --train_valid_split_pkl /home/aa/Downloads/minyang/ --load_pretrained /home/aa/Downloads/minyang/resnet18_best_0.75001.pth --fp16_precision --lr 2e-5 --eta_min 1e-6 --threshold 0.5
	$(MAKE) submit BACKBONE=resnet18 

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
MODEL_PATH=./checkpoints/resnet18/best.pth

submit:
	$(CUDA_ENV) && python3 inference.py --backbone $(BACKBONE) --model_path $(MODEL_PATH) --pred_csv_path ./pred.csv --threshold $(THRESHOLD) --device $(DEVICE)
	python3 to_kaggle.py --pred_csv_path ./pred.csv --out_csv_path ./submit.csv 
	#rm ./pred.csv 

submit_tta:
	python3 inference.py --backbone $(BACKBONE) --model_path $(MODEL_PATH) --pred_csv_path ./pred.csv --threshold $(THRESHOLD) --device $(DEVICE) --bsize 2 --tta --num_tta 5
	python3 to_kaggle.py --pred_csv_path ./pred.csv --out_csv_path ./submit_tta.csv 
	#rm ./pred.csv 



test:
	python3 test.py 

