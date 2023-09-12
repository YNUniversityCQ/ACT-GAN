import argparse

parser = argparse.ArgumentParser(description='Radio map estimate')

# data specifications

parser.add_argument('--image_size', type=int, default=256,
                    help='image size')

# model specifications 
parser.add_argument('--model', type=str, default='au_gan',
                    help='Model name')
parser.add_argument('--block_num', type=int, default=1,
                    help=' AOT numbers')
parser.add_argument('--rates', type=str, default='1+2+4+8',
                    help='Cavity convolution expansion rate')
parser.add_argument('--gan_type', type=str, default='patchgan',
                    help='Discriminator type')

# hardware specifications 
parser.add_argument('--seed', type=int, default=2021,
                    help='Random seed')
parser.add_argument('--num_workers', type=int, default=4,
                    help='cpu num workers')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size')

# optimization specifications 
parser.add_argument('--lrg', type=float, default=1e-5,
                    help='Generator learning rate')
parser.add_argument('--lrd', type=float, default=1e-5,
                    help='Discriminator learning rate')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer type)')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='Optimizer parameter beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='Optimizer parameter beta2')

# loss specifications 
parser.add_argument('--rec_loss', type=str, default='1*MSE+250*Style+0.1*Perceptual',
                    help='Reconstruction loss')
parser.add_argument('--adv_weight', type=float, default=0.01,
                    help='Counter loss weight')

# training specifications 
parser.add_argument('--iterations', type=int, default=40000/8,
                    help='Training data volume')

# Losses are recorded based on the number of iterations
parser.add_argument('--print_every', type=int, default=5,
                    help='Progress bar update')
parser.add_argument('--save_every', type=int, default=5000,
                    help='Training data volume/batch_size * 2)')
parser.add_argument('--save_dir', type=str, default='experiments',
                    help='The directory where the model and logs are stored')
parser.add_argument('--tensorboard', action='store_true',
                    help='default: false, since it will slow training. use it for debugging')

# test
parser.add_argument('--pre_train', type=str, default=None,
                    help='Pre-train the model path')
parser.add_argument('--outputs', type=str, default='outputs',
                    help='Save result path')

# ----------------------------------
args = parser.parse_args()
args.iterations = int(args.iterations)

args.rates = list(map(int, list(args.rates.split('+'))))

losses = list(args.rec_loss.split('+'))
args.rec_loss = {}
for l in losses:
    weight, name = l.split('*')
    args.rec_loss[name] = float(weight)