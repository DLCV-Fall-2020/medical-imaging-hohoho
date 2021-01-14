SHELL := /bin/zsh 

all: pretrained_cnn cnn_lstm_finetune 

################### Stage-1 train backbone ###################

pretrained_cnn:
	cd ./base && make train_resnet18 DEVICE=cuda:0 CH=1

################### Stage-2 tuneing cnnlstm ##################

TRAIN_VALID_SPLIT=../base/checkpoints/resnet18/train_valid_split.pkl
LOAD_MODEL=../base/checkpoints/resnet18/best.pth 

cnn_lstm_finetune:
	cd ./cnnlstm && make train_resnet18_lstm_pretrained_cnn DEVICE=cuda TRAIN_VALID_SPLIT=$(TRAIN_VALID_SPLIT) LOAD_MODEL=$(LOAD_MODEL)
	
	rm -f submit.csv
	cp ./cnnlstm/submit.csv ./

