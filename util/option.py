import argparse
def get_option():
    parser = argparse.ArgumentParser();
    parser.add_argument('--net','-net',type=str,default='BlendedAtlasNet',help='network name')
    parser.add_argument('--batchSize','-bs',type=int,default=32,help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
    parser.add_argument('--nepoch', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--model','-m',type=str,default = '',help='model path load');
    parser.add_argument('--mode','-md', type=str, default = 'AE',  help='mode');
    parser.add_argument('--pts_num', type=int, default = 2500,  help='number of points')
    parser.add_argument('--grid_num', type=int, default = 1,  help='number of grid')
    parser.add_argument('--grid_dim', type=int, default = 3,  help='dim of grid')
    parser.add_argument('--grid_ply', type=str, default = '', help='input grid as mesh in ply file')
    parser.add_argument('--dir', type=str, default = '', help='log dir')
    parser.add_argument('--data', type=str, default = '', help='data path')
    parser.add_argument('--lr',type=float,default=0.001,help='set learning rate');
    parser.add_argument('--weight_decay',type=float,default=1e-9,help='set weight decay');
    parser.add_argument('--gpu','-gpu',type=str,default='0',help='set gpu');
    parser.add_argument('--w','-w',type=float,default=1.0,help='weight for inverse');
    parser.add_argument('--exec','-X',type=str,action='append',help='running tasks');
    parser.add_argument('--manualSeed', type=int, help='seed', default=0)
    parser.add_argument('--topk', type=int, help='blend num k', default=3)
    #binary flags:
    parser.add_argument('--ply',action='store_true');
    return parser.parse_args();