from utils import str2bool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/kaggle/working/DesmokeData/images/dataset', help='Root folder of paired images')
parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data used for validation')
parser.add_argument('--sample_output_folder', type=str, default='samples', help='Validation output image path')
parser.add_argument('--use_gpu', type=str2bool, default=True, help='Use GPU')
parser.add_argument('--gpu', type=int, default=-1, help='GPU id')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--num_workers', type=int, default=2, help='Number of threads for data loader')
parser.add_argument('--print_gap', type=int, default=50, help='number of batches to print average loss')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--val_batch_size', type=int, default=16, help='Validation batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
parser.add_argument('--model_dir', type=str, default='./model')
parser.add_argument('--log_dir', type=str, default='./log')
parser.add_argument('--ckpt', type=str, default='./model/nets/net_1.pkl')
parser.add_argument('--net_name', type=str, default='nets')
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
