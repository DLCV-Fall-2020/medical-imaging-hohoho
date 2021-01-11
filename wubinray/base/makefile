BACKBONE=resnetxx
THRESHOLD=0.5
DEVICE=cuda:1
CH=3

train_resnet18: 
	$(eval BACKBONE=resnet18)
	python3 train.py --backbone $(BACKBONE) --bsize 48 --ch 1 --device $(DEVICE) --fp16_precision
	$(MAKE) submit BACKBONE=resnet18 

train_resnet50:
	$(eval BACKBONE=resnet50)
	python3 train.py --backbone $(BACKBONE) --bsize 48 --ch $(CH) --device $(DEVICE) --fp16_precision
	$(MAKE) submit BACKBONE=resnet50 

submit:
	python3 inference.py --ch $(CH) --backbone $(BACKBONE) --model_path ./checkpoints/resnet18/best.pth --pred_csv_path ./pred.csv --threshold $(THRESHOLD) --device $(DEVICE)
	python3 to_kaggle.py --pred_csv_path ./pred.csv --out_csv_path ./submit.csv 
	rm ./pred.csv 

