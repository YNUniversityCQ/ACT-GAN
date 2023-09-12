import os
import time
import torch
import warnings
import importlib
from glob import glob
from tqdm import tqdm
from model import au_gan
from utils.option import args
from dataloader import loaders
from utils.common import timer
from torchsummary import summary
from loss import loss as loss_module
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":

    device = torch.device("cuda:0")

    # ignore warning
    warnings.filterwarnings("ignore")

    # optimize cuda efficiency
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # load the network and parameters
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.enabled
    model = au_gan.Generator(args)
    model.cuda()

    # scene 1 and scene 3
    summary(model, input_size=[(1, args.image_size, args.image_size),
                               (1, args.image_size, args.image_size)])

    # # scene 2
    # summary(model, input_size=[(1, args.image_size, args.image_size),
    #                            (1, args.image_size, args.image_size),
    #                            (1, args.image_size, args.image_size)])

    class Trainer():
        def __init__(self, args):
            self.args = args
            self.iteration = 0

            # Initial generation loss and antagonistic loss
            self.rec_loss_func = {key: getattr(loss_module, key)() for key, val in args.rec_loss.items()}
            self.adv_loss = getattr(loss_module, args.gan_type)()

            net = importlib.import_module('model.' + args.model)

            # Initialize the network and optimizer
            self.netG = net.Generator(args).cuda()
            self.netD = net.Discriminator().cuda()
            self.optimG = torch.optim.Adam(self.netG.parameters(), lr=args.lrg, betas=(args.beta1, args.beta2))
            self.optimD = torch.optim.Adam(self.netD.parameters(), lr=args.lrd, betas=(args.beta1, args.beta2))
            self.load()

            '''
            Note: '***' Select input scene
                   *** = 1/2/3
            '''

            self.classname = 'AUGAN_scene' + '1'

            # keep a journal
            if args.tensorboard:
                self.writer = SummaryWriter(os.path.join(args.save_dir, 'log'))

            # data loading
            # generator=torch.Generator(device='cuda')
            self.AUGAN_train = getattr(loaders, self.classname)(phase='train')
            self.dataloaders = DataLoader(self.AUGAN_train, shuffle=True, pin_memory=True,
                                          batch_size=args.batch_size, num_workers=args.num_workers)

        # Load the parameters before training
        def load(self):
            try:
                # Generator weight parameter path
                gpath = sorted(list(glob(os.path.join(self.args.save_dir, 'G*.pt'))))[-1]
                self.netG.load_state_dict(torch.load(gpath, map_location='cuda'))
                self.iteration = int(os.path.basename(gpath)[1:-3])
                print(f'[**] Loading generator network from {gpath}')
            except:
                pass

            try:
                # Discriminator weight parameter path
                dpath = sorted(list(glob(os.path.join(self.args.save_dir, 'D*.pt'))))[-1]
                self.netD.load_state_dict(torch.load(dpath, map_location='cuda'))
                print(f'[**] Loading discriminator network from {dpath}')
            except:
                pass

            try:
                # Optimizer weight parameter path
                opath = sorted(list(glob(os.path.join(self.args.save_dir, 'O*.pt'))))[-1]
                data = torch.load(opath, map_location='cuda')
                self.optimG.load_state_dict(data['optimG'])
                self.optimD.load_state_dict(data['optimD'])
                print(f'[**] Loading optimizer from {opath}')
            except:
                pass

        # Save weight parameter
        def save(self, ):

            print(f'\nsaving {self.iteration} model to {self.args.save_dir} ...')

            # generator
            torch.save(self.netG.state_dict(),
                       os.path.join(self.args.save_dir, f'G{str(self.iteration).zfill(7)}.pt'))

            # discriminator
            torch.save(self.netD.state_dict(),
                       os.path.join(self.args.save_dir, f'D{str(self.iteration).zfill(7)}.pt'))
            # optimizer
            torch.save(
                {'optimG': self.optimG.state_dict(), 'optimD': self.optimD.state_dict()},
                os.path.join(self.args.save_dir, f'O{str(self.iteration).zfill(7)}.pt'))

        # define training
        def train(self, num_epoch=100):

            timer_data, timer_model = timer(), timer()
            for epoch in range(num_epoch):
                print('now epoch', epoch + 1)

                since = time.time()
                loop = tqdm(self.dataloaders)
                # scene 1
                for build, antenna, target, img_name in loop:
                # scene 2
                # for build, antenna, sample, target, img_name in loop:
                # scene 3
                # for sample, mask, target, img_name in loop:

                    self.iteration += 1

                    # scene 1
                    builds, antennas, targets = build.cuda(), antenna.cuda(), target.cuda()
                    # scene 2
                    # builds, antennas, samples, targets = build.cuda(), antenna.cuda(), sample.cuda(), target.cuda()
                    # scene 3
                    # samples, masks, targets = sample.cuda(), mask.cuda(), target.cuda()

                    # scene 1
                    predict_image = self.netG(builds, antennas)
                    # scene 2
                    # predict_image = self.netG(builds, antennas, samples)
                    # scene 3
                    # predict_image = self.netG(samples, masks)

                    # reconstruction loss
                    losses = {}

                    # L1, perception, style names and weights
                    for name, weight in self.args.rec_loss.items():
                        losses[name] = weight * self.rec_loss_func[name](predict_image, targets)

                    # counter loss
                    dis_loss, gen_loss = self.adv_loss(self.netD, predict_image, targets)
                    losses[f"advg"] = gen_loss * self.args.adv_weight

                    # Gradient zeroing, backpropagation, updating optimizer parameters
                    self.optimG.zero_grad()
                    self.optimD.zero_grad()
                    sum(losses.values()).backward()
                    losses[f"advd"] = dis_loss
                    dis_loss.backward()
                    self.optimG.step()
                    self.optimD.step()

                    timer_model.hold()
                    timer_data.tic()
                    description = f'mt:{timer_model.release():.1f}s, dt:{timer_data.release():.1f}s, '
                    for key, val in losses.items():
                        description += f'{key}:{val.item():.6f}, '
                        if self.args.tensorboard:
                            self.writer.add_scalar(key, val.item(), self.iteration)
                    loop.set_description((description))

                    # Save the weight by number of iterations
                    if self.iteration % self.args.save_every == 0:
                        self.save()


                time_elapsed = time.time() - since
                print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    trainer = Trainer(args)
    trainer.train()

