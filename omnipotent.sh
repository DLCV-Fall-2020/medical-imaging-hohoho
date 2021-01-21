wget -O hemo_resnet18_0.75001.pth 'https://www.dropbox.com/s/bwr6ccp65vm9lqo/hemo_resnet18_0.75001.pth?dl=1'
wget -O hemo_lstm_best_0.78162.pth 'https://www.dropbox.com/s/jrar6yzietv72z4/hemo_lstm_best_0.78162.pth?dl=1'

python3 preprocessing.py --data $1 --save test_embedding.pkl
python3 inference.py --data $1 --embedding test_embedding.pkl --output_csv $2