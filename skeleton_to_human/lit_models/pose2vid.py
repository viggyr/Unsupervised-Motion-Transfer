from skeleton_to_human import models, options
import torch.nn as nn
import torch
from ..models import networks_modified as networks
import argparse
try:
    import wandb
except ModuleNotFoundError:
    pass


#from .metrics import CharacterErrorRate
from .base import BaseLitModel


class Pose2Vid(BaseLitModel):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    The module must take x, y as inputs, and have a special predict() method.
    """

    def __init__(self, model,  args: argparse.Namespace = None):
        super().__init__(model, args)
        self.no_vgg_loss=self.args.get("no_vgg_loss", False)
        self.no_flow_loss=self.args.get("no_flow_loss", False)
        self.no_gan_feat_loss=self.args.get("no_ganFeat_loss", False)
        self.no_lsgan = self.args.get("no_lsgan", False)
        self.dataroot = self.args.get("dataroot", './datasets/cityscapes/')
        self.criterionGAN = networks.GANLoss(use_lsgan=not self.no_lsgan, tensor=self.model.Tensor)   
        self.criterionFeat = torch.nn.L1Loss()
        if not self.no_vgg_loss:             
            self.criterionVGG = networks.VGGLoss()

        if not self.no_flow_loss:
            # 20181013 Flow L1 needs averaging
            self.nelem = 288*512 if self.dataroot.find('512') != -1 else 576*1024
            # print(self.nelem / 512) 
            self.criterionFlow = networks.FlowLoss()
        
        # Discriminator network
        use_sigmoid = self.args.get("no_lsgan", False)
        netD_input_nc = self.args.get("input_nc",3) + self.args.get("output_nc",3)
        if not self.args.get("no_instance",True):
            netD_input_nc += 1

        # TODO: 20180929: Generator Input contains two images...
        netD_input_nc *= 2  # two pairs of pose/frame
        self.netD = networks.define_D(netD_input_nc, self.args.get("ndf"), self.args.get("n_layers_D"), self.args.get("norm"), use_sigmoid, 
                                        self.args.get("num_D"), not self.args.get("no_ganFeat_loss"))
            
        #Todo: Visualization.
        #Todo: Setup test and validation metrics and functions.
        #self.val_cer = CharacterErrorRate(ignore_tokens)
        #self.test_cer = CharacterErrorRate(ignore_tokens)

    def forward(self, x):
        return self.model.predict(x)

    def training_step(self, data, batch_idx, optimizer_idx):
        label = data['label'].data.cuda()
        real_image = data['image'].data.cuda()
        gt1 = real_image[:, 0, ...]
        gt2 = real_image[:, 1, ...]
        ### Generator Forward, predict two frames
        x1 = label[:, 0, ...]
        x2 = label[:, 1, ...]
        y1, y2 = self.model(x1, x2)
        losses=[]
        # Real Detection and Loss        
        pred_real = self.netD.forward(torch.cat((x1, x2, gt1.detach(), gt2.detach()), dim=1))
        if optimizer_idx==0:
            # GAN loss (Fake Possibility Loss)
            pred_fake = self.netD.forward(torch.cat((x1, x2, y1, y2), dim=1))
            losses.append(self.criterionGAN(pred_fake, True))

            # GAN feature matching loss
            loss_G_GAN_Feat = 0
            if not self.no_gan_feat_loss:
                feat_weights = 4.0 / (self.args.get("n_layers_D",3) + 1)
                D_weights = 1.0 / self.opt.num_D
                for i in range(self.opt.num_D):
                    for j in range(len(pred_fake[i])-1):
                        loss_G_GAN_Feat += D_weights * feat_weights * \
                            self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.lambda_feat
            
            losses.append(loss_G_GAN_Feat)
            # VGG feature matching loss
            loss_G_VGG = 0
            if not self.opt.no_vgg_loss:
                loss_G_VGG = (self.criterionVGG(y1, gt1) + self.criterionVGG(y2, gt2))\
                            * self.opt.lambda_feat
            losses.append(loss_G_VGG)
             # # 20181012: pwc-flow matching loss
            loss_G_flow = 0
            if not self.opt.no_flow_loss:
                loss_G_flow = self.criterionFlow(gt1.detach(), gt2.detach(), y1, y2) * self.opt.lambda_flow / self.nelem
            losses.append(loss_G_flow)
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            #20180930: Always return fake_B now, let super function decide whether to save it  
            self.log('train_generator_loss', sum(losses))            
            return sum(losses)
        else:
            # Fake Detection and Loss
            pred_fake_pool = self.netD.forward(torch.cat((x1, x2, y1.detach(), y2.detach()), dim=1))
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)       
            losses.append(loss_D_fake)
            
            loss_D_real = self.criterionGAN(pred_real, True)
            losses.append(loss_D_real)
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            self.log('train_discriminator_loss', sum(losses)*0.5)       
            return sum(losses) * 0.5
        # Todo: What about returning the generated images? For save_fake argument in original repo.

        # y1_clean = torch.squeeze(y1.detach())
        # y2_clean = torch.squeeze(y2.detach())

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        pass

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        pass
