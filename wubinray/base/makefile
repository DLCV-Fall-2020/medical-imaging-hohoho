
train:
	python3 train.py --bsize 64 --ch 3 --device cuda:1

submit_test:
	python3 inference.py --model_path ./checkpoints/resnet18/best.pth --pred_csv_path ./pred.csv --threshold 0.5 --device cuda:0
	python3 to_kaggle.py --pred_csv_path ./pred.csv --out_csv_path ./submit.csv 
	rm ./pred.csv 

