import os
import importlib
from tqdm import tqdm
from glob import glob

import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from data import create_loader 
from loss import loss as loss_module
from .common import timer, reduce_loss_dict

class Trainer():
    def __init__(self, args):
        self.args = args 
        self.iteration = 0

        # 设置数据集和数据加载器
        self.dataloader = create_loader(args)

        # 设置损失和指标
        # key：单个损失函数名称  val：损失函数比重
        self.rec_loss_func = {
            key: getattr(loss_module, key)() for key, val in args.rec_loss.items()}

        # 此处调用了smgan的高斯平均损失函数
        self.adv_loss = getattr(loss_module, args.gan_type)()

        # 生成器输入(四通道): [rgb(3) + mask(1)], 判别器输入(三通道): [rgb(3)]
        net = importlib.import_module('model.'+args.model)

        # 生成器网络
        self.netG = net.InpaintGenerator(args).cuda()
        # 生成器优化器
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), lr=args.lrg, betas=(args.beta1, args.beta2))
        # 判别器网络
        self.netD = net.Discriminator().cuda()
        # 判别器优化器
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), lr=args.lrd, betas=(args.beta1, args.beta2))
        
        self.load()
        # 多卡训练
        if args.distributed:
            self.netG = DDP(self.netG, device_ids= [args.local_rank], output_device=[args.local_rank])
            self.netD = DDP(self.netD, device_ids= [args.local_rank], output_device=[args.local_rank])
        # 写日志
        if args.tensorboard: 
            self.writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
            

    def load(self):
        try:
            # 生成器权重参数路径
            gpath = sorted(list(glob(os.path.join(self.args.save_dir, 'G*.pt'))))[-1]
            self.netG.load_state_dict(torch.load(gpath, map_location='cuda'))
            self.iteration = int(os.path.basename(gpath)[1:-3])
            if self.args.global_rank == 0: 
                print(f'[**] Loading generator network from {gpath}')
        except: 
            pass 
        
        try:
            # 判别器权重参数路径
            dpath = sorted(list(glob(os.path.join(self.args.save_dir, 'D*.pt'))))[-1]
            self.netD.load_state_dict(torch.load(dpath, map_location='cuda'))
            if self.args.global_rank == 0: 
                print(f'[**] Loading discriminator network from {dpath}')
        except: 
            pass
        
        try:
            # 优化器参数？？？母鸡
            opath = sorted(list(glob(os.path.join(self.args.save_dir, 'O*.pt'))))[-1]
            data = torch.load(opath, map_location='cuda')
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            if self.args.global_rank == 0: 
                print(f'[**] Loading optimizer from {opath}')
        except: 
            pass

    # 保存权重参数
    def save(self, ):
        if self.args.global_rank == 0:
            print(f'\nsaving {self.iteration} model to {self.args.save_dir} ...')
            torch.save(self.netG.state_dict(),
                os.path.join(self.args.save_dir, f'G{str(self.iteration).zfill(7)}.pt'))
            torch.save(self.netD.state_dict(),
                os.path.join(self.args.save_dir, f'D{str(self.iteration).zfill(7)}.pt'))
            torch.save(
                {'optimG': self.optimG.state_dict(), 'optimD': self.optimD.state_dict()}, 
                os.path.join(self.args.save_dir, f'O{str(self.iteration).zfill(7)}.pt'))
            
    # 自定义训练
    def train(self):
        pbar = range(self.iteration, self.args.iterations)


        # 训练计时器
        if self.args.global_rank == 0: 
            pbar = tqdm(range(self.args.iterations), initial=0, dynamic_ncols=True, smoothing=0.01)

            timer_data, timer_model = timer(), timer()

        # 嵌套式循环(大)

        for _ in pbar:
            self.iteration += 1
            images, masks, builds, filename = next(self.dataloader)
            # 分别为torch.size[2,3,515,512] and torch.size[2,1,515,512]  注：(原图与掩码)
            # images, masks, builds = images.cuda(), masks.cuda(), builds.cuda()
            images, masks, builds = images.cuda(), masks.cuda(), builds.cuda()
            # torch.size[2,3,515,512]   注：(原图上挖洞)
            images_masked = (images * (1 - masks).float()) + masks



            if self.args.global_rank == 0: 
                timer_data.hold()
                timer_model.tic()

            # 输入: [rgb(3) + edge(1)]
            # pred_img = self.netG(images_masked, masks, builds)
            pred_img = self.netG(images_masked, masks, builds)

            # 原图上挖洞的图与预测图反向挖洞的图相合并
            comp_img = (1 - masks) * images + masks * pred_img


            # 重构损失
            losses = {}
            # L1、感知、风格的名称和比重
            for name, weight in self.args.rec_loss.items(): 
                losses[name] = weight * self.rec_loss_func[name](pred_img, images)
            
            # 对抗损失
            dis_loss, gen_loss = self.adv_loss(self.netD, pred_img, images, masks)
            losses[f"advg"] = gen_loss * self.args.adv_weight
            
            # 反向传播
            self.optimG.zero_grad()
            self.optimD.zero_grad()
            sum(losses.values()).backward()
            losses[f"advd"] = dis_loss 
            dis_loss.backward()
            self.optimG.step()
            self.optimD.step()



            if self.args.global_rank == 0:
                timer_model.hold()
                timer_data.tic()

            # logs
            scalar_reduced = reduce_loss_dict(losses, self.args.world_size)
            if self.args.global_rank == 0 and (self.iteration % self.args.print_every == 0): 
                pbar.update(self.args.print_every)
                description = f'mt:{timer_model.release():.1f}s, dt:{timer_data.release():.1f}s, '
                for key, val in losses.items(): 
                    description += f'{key}:{val.item():.3f}, '
                    if self.args.tensorboard: 
                        self.writer.add_scalar(key, val.item(), self.iteration)
                pbar.set_description((description))
                if self.args.tensorboard: 
                    self.writer.add_image('mask', make_grid(masks), self.iteration)
                    self.writer.add_image('orig', make_grid((images+1.0)/2.0), self.iteration)
                    self.writer.add_image('pred', make_grid((pred_img+1.0)/2.0), self.iteration)
                    self.writer.add_image('comp', make_grid((comp_img+1.0)/2.0), self.iteration)
                    
            
            if self.args.global_rank == 0 and (self.iteration % self.args.save_every) == 0: 
                self.save()


