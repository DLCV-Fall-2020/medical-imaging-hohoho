import argparse

def parse_args(string=None):
    parser = argparse.ArgumentParser(description='Blood_Base')
    # train args
    parser.add_argument('--bsize', type=int, default=8,
                        help="batch size")
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help="lr warmup epochs")
    parser.add_argument('--epochs', type=int, default=100,
                        help="epochs")
    parser.add_argument('--num_workers', type=int, default=5,
                        help="dataloader workers")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="cpu, cuda:0, cuda:1, .....")
    parser.add_argument('--fp16_precision', action="store_true",
                        help="train on fp16")
    # dataset
    parser.add_argument('--ch', type=int, default=1,
                        help="input channel, how many img to one sample")
    parser.add_argument('--img_size', type=int, default=512,
                        help="input img size")
    parser.add_argument('--valid_size', type=float, default=0.15,
                        help="valid split %")
    parser.add_argument('--num_classes', type=int, default=5,
                        help="dataset classes")
    # model
    parser.add_argument('--backbone', type=str, default='resnet10_3d',
                        choices=["resnet10_3d", "resnet18_3d"],
                        help='backbone used')
    parser.add_argument('--pretrained', action="store_true")
    parser.add_argument('--hidd_dim', type=int, default=128,
                        help='resnet3d output dimension')
    parser.add_argument('--proj_dim', type=int, default=32,
                        help="project head projected dimension")
    # loss
    parser.add_argument('--use_cos_similarity', action="store_true",
                        help="calculate loss use cosine similairty")
    parser.add_argument('--temperature', type=float, default=0.07, #0.07 0.5
                        help="NT-Xent loss temperature")
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument('--eta_min', type=float, default=2e-4,
                        help="cosin annealing to lr=eta_min")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="LARS optimizer weight_decay")


    if string is not None: args = parser.parse_args(string)  
    else: args = parser.parse_args()

    return args 
