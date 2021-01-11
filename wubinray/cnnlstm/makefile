BACKBONE=resnetxx_3d
THRESHOLD=0.5
DEVICE=cuda 
BSIZE=48
BISZE2=16

train_resnet18_lstm: 
	$(eval BACKBONE=resnet18)
	python3 train.py --backbone $(BACKBONE) --bsize $(BSIZE) --t 5 --device $(DEVICE) --fp16_precision
	$(MAKE) submit BACKBONE=resnet18 

train_resnet34_lstm:
	$(eval BACKBONE=resnet34)
	python3 train.py --backbone $(BACKBONE) --bsize $(BSIZE2) --device cuda --fp16_precision 
	$(MAKE) submit BACKBONE=resnet34 

train_resnet50_lstm:
	$(eval BACKBONE=resnet50)
	python3 train.py --backbone $(BACKBONE) --bsize 16 --device cuda --fp16_precision 

submit:
	python3 inference.py --backbone $(BACKBONE) --model_path ./checkpoints/resnet18/best.pth --pred_csv_path ./pred.csv --threshold $(THRESHOLD) --device cuda:1
	python3 to_kaggle.py --pred_csv_path ./pred.csv --out_csv_path ./submit.csv 
	rm ./pred.csv 

test:
	python3 test.py 

