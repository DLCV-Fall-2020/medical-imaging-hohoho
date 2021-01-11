BACKBONE=resnetxx_3d
THRESHOLD=0.5

train_resnet10_3d: 
	$(eval BACKBONE=resnet10_3d)
	python3 train.py --backbone $(BACKBONE) --bsize 16 --device cuda:0
	$(MAKE) submit BACKBONE=resnet10_3d

train_resnet18_3d:
	$(eval BACKBONE=resnet18_3d)
	python3 train.py --backbone $(BACKBONE) --bsize 16 --device cuda:0
	$(MAKE) submit BACKBONE=resnet18_3d

submit:
	python3 inference.py --backbone $(BACKBONE) --model_path ./checkpoints/resnet18/best.pth --pred_csv_path ./pred.csv --threshold $(THRESHOLD) --device cuda:1
	python3 to_kaggle.py --pred_csv_path ./pred.csv --out_csv_path ./submit.csv 
	rm ./pred.csv 

