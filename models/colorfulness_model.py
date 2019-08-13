import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import resnet
import sys
from torch.utils.serialization import load_lua
import torch.nn as nn
import torchvision
import random

class Colorfulness_Model(BaseModel):
    def name(self):
        return 'Colorfulness_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize 

        self.netG_A = networks.define_G(3, 3,opt.ngf, 'resonly', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_A.cuda()
        #self.netG_A = resnet.ResNet(depth=26,num_classes=5)
        #self.netF = networks.define_F(self.gpu_ids, use_bn=False)
        self.loss=0.0
        self.correct= 0
        self.wrong = 0

        self.statistics= np.ones(5)
        self.statistics_acc = np.zeros(5)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            #self.netD_B = networks.define_D(opt.input_nc, opt.ndf,opt.which_model_netD,opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
          
            self.load_network(self.netG_A, 'G_A', which_epoch)

            #if self.isTrain:
                #self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            #self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionCrossEntropy = torch.nn.CrossEntropyLoss().cuda()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                                 self.netG_A.parameters(),
                                                 ),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.SGD(itertools.chain(
                                                 self.netG_A.parameters(),
                                                 ),lr=opt.lr, momentum=0.999)
            #self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            #self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)


        #if self.isTrain:
        #    networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['In']
        input_label = input['Expert']
      

       
        #if len(self.gpu_ids) > 0:
        #    input_A = input_A.cuda(self.gpu_ids[0], async=True)
            #input_B = input_B.cuda(self.gpu_ids[0], async=True)

        self.input_A = input_A
        self.input_label = self.to_tensor_label(input_label)
        self.lab= input_label

        #self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def to_tensor_label(self,label):
    	a = np.zeros(5)
    	a[label]=1
    	b= torch.from_numpy(a).unsqueeze(0)
    	return b


    def forward(self):
        self.real_A = Variable(self.input_A)

    def test(self):

        real_A = Variable(self.input_A)
        lab = Variable(self.lab)

        real_A = real_A.float().cuda()

        prediction = self.netG_A(real_A)
        
        #print("XXXXXXXXXXX")
        #print(prediction)
        value, indices = torch.max(prediction,1)
        #print(value)
        #print(indices)
        #print(lab)
        self.statistics[lab.long().cuda()] = self.statistics[lab.long().cuda()] + 1


        if indices == lab.long().cuda() :
            self.correct = self.correct + 1
            self.statistics_acc[indices] = self.statistics_acc[indices] + 1
        else:
            self.wrong = self.wrong + 1

        ratio = self.correct / (self.correct + self.wrong)
        print(ratio)
        print(self.statistics_acc)
        print(self.statistics)
        print(self.statistics_acc/self.statistics)


    def colorfulness_metric(self,img):
        img = image_torch


        rg = img[0] - img[1]
        yb = (img[0] - img[1])*0.5 - img[2]

        mean_rg = rg.view(-1).mean(dim=0)
        mean_yb = yb.view(-1).mean(dim=0)
        
        var_rg = rg.view(-1).std(dim=0)
        var_yb = yb.view(-1).std(dim=0)
        
        total_var = (var_rg**2 + var_yb**2)**0.5
        total_mean = (mean_rg**2 + mean_yb**2)**0.5
        
        colorfulness = total_var + 0.3*total_mean

        normalized_colorfulness = colorfulness / 150 * 10  # /max_value(assumed) * max_value()

        return colorfulness

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    #def backward_D_basic(self, netD, real, fake):
    #    # Real
    #    pred_real = netD(real.float())
    #    loss_D_real = self.criterionGAN(pred_real, True)
    #    # Fake
    #    pred_fake = netD(fake.float().detach())
    #    loss_D_fake = self.criterionGAN(pred_fake, False)
    #    # Combined loss
    #    loss_D = (loss_D_real + loss_D_fake) * 0.5
    #    # backward
    #    loss_D.backward()
    #    return loss_D

    #def backward_D_A(self):
    #    fake_A = self.fake_A_pool.query(self.fake_A)
    #    loss_D_A = self.backward_D_basic(self.netD_A, self.real_A_gray, fake_A)
    #    self.loss_D_A = loss_D_A.data[0]

    #def backward_D_B(self):
    #    fake_B = self.fake_B_pool.query(self.fake_B)
    #    loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B)
    #    self.loss_D_B = loss_D_B.data[0]

    #def backward_D_seg(self,netD_seg,real_seg,fake_seg):
    #    loss_D_seg = self.backward_D_basic(netD_seg, real_seg, fake_seg)
    #    return loss_D_seg.data[0]
   

    def backward_all(self):

        prediction = self.netG_A(self.real_A.cuda().float())
        label = self.colorfulness_metric(self.real_A.cuda().float())
        #label = self.input_label.float().cuda()

        #print(prediction)
        #print(label)

        print("XXXXXXXXXXX")
        print(label)
        print(prediction)
        loss_G = self.criterionCrossEntropy(prediction, self.lab.long().cuda())
        loss_G.backward()
        self.loss_G = loss_G

        print(self.loss_G)        
        

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.forward()
        self.backward_all()
        self.optimizer_G.step()

        #print('Loss: %d' % (self.loss_G.item()))




    def get_current_errors(self): 
        ret_errors = OrderedDict([
            ('loss_G', self.loss_G),
            ])
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A,False)
        #real_B = util.tensor2im(self.real_B,False)

        #fake_B = util.tensor2im(self.fake_B, False)


        ret_visuals = OrderedDict([('real_A', real_A),
                                   #('fake_B', fake_B),
                                   #('real_B', real_B),
         ])
        return ret_visuals


    def get_current_visuals_test(self):
        real_A = util.tensor2im(self.real_A,False)
        #fake_B = util.tensor2im(self.fake_B, False)
        
        ret_visuals = OrderedDict([('real_A', real_A), 
                                   #('fake_B', fake_B),
         ])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        #self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
