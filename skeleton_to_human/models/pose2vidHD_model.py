### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from .base_model import BaseModel
# from .net import networks
from . import networks_modified as networks

# TODO: modify upon pix2pixHDModel, adding last-frame reference
#   TODO: generator, input two images now
#   TODO: discriminator, using conv2D but adding PWC-net for optical flow alignment as well
#   TODO: loss, using both feature maching and VGG, also implement PWC-loss in the future.

class Pose2VidHDModel(torch.nn.Module):
    def name(self):
        return 'Pose2VidHDModel'
    
    def __init__(self, opt):
        super().__init__()
        #BaseModel.initialize(self, opt)
        self.opt = opt
        #self.gpu_ids = opt.gpu_ids
        #self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if opt.gpus else torch.Tensor
        #self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        # self.use_features = opt.instance_feat or opt.label_feat
        # self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        if not opt.no_instance:
            netG_input_nc += 1
        # if self.use_features:
        #     netG_input_nc += opt.feat_num

        # TODO: 20180929: Generator Input contains two images...
        netG_input_nc += opt.output_nc  # also contains the previous frame
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm)        

        

        ### Encoder network
        # if self.gen_features:          
        #     self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
        #                                   opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
        

        # Todo: continue training from checkpoint.
        # if opt.continue_train or opt.load_pretrain:
        #     pretrained_path = '' if not self.isTrain else opt.load_pretrain
        #     self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
        #     if self.isTrain:
        #         self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
            # if self.gen_features:
            #     self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

       

       

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):             
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1) 
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # # instance map for feature encoding
        # if self.use_features:
        #     # get precomputed feature maps
        #     if self.opt.load_features:
        #         feat_map = Variable(feat_map.data.cuda())

        return input_label, inst_map, real_image, feat_map

    # TODO: 20180930: Ignore features for now.
    def forward(self, x1, x2):
        #Error? - Check for zero dimensions. I changed it to x1 instead of gt1.
        zero = torch.zeros_like(x1)
        y1 = self.netG.forward(torch.cat((x1, zero), 1))
        y2 = self.netG.forward(torch.cat((x2, y1), 1))
        return y1, y2

    def predict(self, label, inst, prev_frame):
        # Encode Inputs        
        input_label, inst_map, _, _ = self.encode_input(Variable(label), Variable(inst), infer=True)
        if self.opt.gpus:
            prev_frame = Variable(prev_frame.data.cuda())

        # # Fake Generation
        # if self.use_features:       
        #     # sample clusters from precomputed features             
        #     feat_map = self.sample_features(inst_map)
        #     input_concat = torch.cat((input_label, feat_map), dim=1)                        
        # else:
        input_concat = input_label

        input_concat = torch.cat((input_concat, prev_frame), dim=1)

        # TODO：Write a for loop that iteratively generates each frame
        # TODO: Also provide a option as whether to provide the first ground-truth frame
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
       

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    @staticmethod
    def add_to_argparse(parser):
        
        
        # input/output sizes       
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        parser.add_argument('--label_nc', type=int, default=0, help='# of input label channels')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

         # for generator
        parser.add_argument('--netG', type=str, default='local', help='selects model to use for netG')
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        

        parser.add_argument('--resize_or_crop', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')


class InferenceModel(Pose2VidHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.predict(label, inst)
