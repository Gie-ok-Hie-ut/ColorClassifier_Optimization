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

class ColorClassify_Model2(BaseModel):
    def name(self):
        return 'ColorClassify_Model2'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize

        self.num_last_class= 2 # 0 ~ num_class-1
        self.interest_class = 0

        #self.netG_A = networks.define_G(3, 3,opt.ngf, 'resonly', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        #self.netG_A.cuda()
        #self.netG_A = networks.define_pretrained('vgg19') # 'vgg19'
        #self.netG_A = networks.define_F(self.gpu_ids,extract=True,use_input_norm=False,use_bn=False,colorcheckpoint=True)
        #self.netF_A = networks.define_F(self.gpu_ids,extract=True,use_input_norm=True,use_bn=False,colorcheckpoint=False)
        #self.netG_A = networks.define_pretrained('nasnetalarge') # 'vgg19'


        self.netG_A = networks.define_VGG()
        self.netG_A.cuda()

        #self.netC_A = networks.define_C(1, 64, opt.ngf, 'basic', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids) # out_nc=32
        print(self.netG_A)
        print(self.netG_A.state_dict().keys())        
        
        self.netAdhoc = networks.define_adhoc2(self.num_last_class)
        self.netAdhoc.cuda()


        # Layers
        self.layers_last = ['p5']
        self.layers_extract = ['r21','r31','r41', 'r51'] # Extract

        #self.string_centroid_r21 = './centroid/Syn7_ep10/centroid_Syn_night_r21.pt'
        #self.string_centroid_r31 = './centroid/Syn7_ep10/centroid_Syn_night_r31.pt'
        #self.string_centroid_r41 = './centroid/Syn7_ep10/centroid_Syn_night_r41.pt'
        #self.string_centroid_r51 = './centroid/Syn7_ep10/centroid_Syn_night_r51.pt'

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            #self.netD_B = networks.define_D(opt.input_nc, opt.ndf,opt.which_model_netD,opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)


        if not self.isTrain: #Test or Extract
            self.loss=0.0
            self.correct= 0
            self.wrong = 0
            self.statistics= np.ones(self.num_last_class)
            self.statistics_acc = np.zeros(self.num_last_class)

            self.correct_cent = 0
            self.wrong_cent = 0
            self.statistics_cent= np.ones(self.num_last_class)
            self.statistics_acc_cent = np.zeros(self.num_last_class)

        self.netG_A.load_state_dict(torch.load('./vgg_original.pth')) #original vgg
        
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
          
            
            #self.netG_A.load_state_dict(torch.load('./15_net_G_A.pth')) #original vgg
            self.netG_A.load_state_dict(torch.load('./vgg_original.pth')) #original vgg
            #self.load_network(self.netG_A, 'G_A', which_epoch)
            #self.load_network(self.netAdhoc,'Adhoc',which_epoch)
            #self.load_network(self.netF,'vgg19',which_epoch)
            #self.netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))

            

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
                                                 self.netAdhoc.parameters(),
                                                 ),lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            #self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netAdhoc)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['In']
        input_label = input['Label']
      
        self.input_A = input_A
        self.input_label = self.to_tensor_label(input_label)
        self.lab= input_label

    def to_tensor_label(self,label):
        a = np.zeros(self.num_last_class)
        a[label]=1
        b= torch.from_numpy(a).unsqueeze(0)
        return b

    def set_input_soft(self, input):
        input_A = input['In']
        input_label = input['Expert']
        input_A2 = input['In2']
        input_label2 = input['Expert2']
      
        self.input_A = input_A
        self.input_A2 = input_A2
        self.input_label = self.to_tensor_label_soft(input_label,input_label2)
        self.lab= input_label

    def to_tensor_label_soft(self,label,label2):
        a = np.zeros(self.num_last_class)


        if label == label2:
            a[label]=1
        else:
            a[label]=0.5
            a[label2]=0.5

        b= torch.from_numpy(a).unsqueeze(0)
        return b

    def getHistogram1d_np(self,img_torch, num_bin): # L space # Idon't know why but they(np, conv) are not exactly same
        # Preprocess
        arr = np.asarray(img_torch)
        arr0 = ( arr[0][0].ravel()[np.flatnonzero(arr[0][0])] + 1 ) / 2 
        arr1 = np.zeros(arr0.size)

        arr_new = [arr0, arr1]
        H, edges = np.histogramdd(arr_new, bins = [num_bin, 1], range =((0,1),(-1,2)))

        H_torch = torch.from_numpy(H).float().cuda() #10/224/224
        H_torch = H_torch.unsqueeze(0).unsqueeze(0).permute(0,2,1,3)
        # Normalize

        total_num = sum(sum(H_torch.squeeze(0).squeeze(0))) # 256 * 256 => same value as arr[0][0].ravel()[np.flatnonzero(arr[0][0])].shape
        H_torch = H_torch / total_num

        return H_torch

    def getHistogram2d_np(self, img_torch, num_bin): # AB space # num_bin = self.hist_ab = 64
        # Preprocess
        #print("NAEGAE")
        #print(img_torch.size())
        arr = np.asarray(img_torch)

        # Exclude Zeros and Make value 0 ~ 1
        arr1 = ( arr[0][1].ravel()[np.flatnonzero(arr[0][1])] + 1 ) /2 
        arr2 = ( arr[0][2].ravel()[np.flatnonzero(arr[0][2])] + 1 ) /2 


        if (arr1.shape[0] != arr2.shape[0]):
            arr2 = np.concatenate([arr2, np.array([0])])
            print("Histogram Size Not Match!")

        #print(arr1.shape[0])
        #print(arr2.shape[0])


        
        #print(arr[0][1].ravel()[np.flatnonzero(arr[0][1])].shape)
        #print(arr[0][2].ravel()[np.flatnonzero(arr[0][2])].shape)

        # AB space
        arr_new = [arr1, arr2]
        #print(arr_new)
        H,edges = np.histogramdd(arr_new, bins = [num_bin, num_bin], range = ((0,1),(0,1)))


        H = np.rot90(H)
        H = np.flip(H,0)

        H_torch = torch.from_numpy(H).float().cuda() #10/224/224
        H_torch = H_torch.unsqueeze(0).unsqueeze(0)

        # Normalize
        total_num = sum(sum(H_torch.squeeze(0).squeeze(0))) # 256 * 256 => same value as arr[0][0].ravel()[np.flatnonzero(arr[0][0])].shape
        H_torch = H_torch / total_num

        return H_torch #1/1/64/64


    def forward(self):
        self.real_A = Variable(self.input_A)
        #self.real_A2 = Variable(self.input_A2)

    def test(self):

        real_A = Variable(self.input_A)
        lab = Variable(self.lab)

        real_A = real_A.float().cuda()

        
        prediction1 = self.netG_A(real_A,self.layers_last)
        print(prediction1)

        prediction2 = self.netAdhoc(prediction1)

        #print(prediction)
        
        #print("XXXXXXXXXXX")
        #print(prediction)
        value, indices = torch.max(prediction2,1)
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
   
    def backward_all_soft(self):
        #prediction = self.netG_A(self.real_A.cuda().float())
        prediction_A1 = self.netG_A(self.real_A.cuda().float(),self.layers_last)
        prediction_A2 = self.netG_A(self.real_A2.cuda().float(),self.layers_last)

        #prediction_mean = (prediction_A1[0] + prediction_A2[0])/2
        prediction_mean = (torch.stack(prediction_A1) + torch.stack(prediction_A2))/2
        #print(torch.stack(prediction_A1).shape)
        prediction_mean = prediction_mean.squeeze(0)
        #print(prediction_mean.squeeze(0))
        #print(k)


        prediction2 = self.netAdhoc([prediction_mean])

        label = self.input_label.long().cuda()

        #label = self.input_label.float().cuda()

        #print(prediction)
        #print(label)

        print("XXXXXXXXXXX")
        print(label)
        print(prediction2)
        loss_G = self.criterionCrossEntropy(prediction2, self.lab.long().cuda())
        loss_G.backward()
        self.loss_G = loss_G

        print(self.loss_G) 

    def backward_all(self):
        #prediction = self.netG_A(self.real_A.cuda().float())

        prediction1 = self.netG_A(self.real_A.cuda().float(),self.layers_last)


        hist_A_real_ab = self.getHistogram2d_np(self.real_A, self.hist_ab)


        prediction2 = self.netAdhoc(prediction1)
        label = self.input_label.long().cuda()

        #label = self.input_label.float().cuda()

        #print(prediction)
        #print(label)

        print("XXXXXXXXXXX")
        print(label)
        print(prediction2)
        loss_G = self.criterionCrossEntropy(prediction2, self.lab.long().cuda())
        loss_G.backward()
        self.loss_G = loss_G

        print(self.loss_G)   


    def backward_all(self):
        #prediction = self.netG_A(self.real_A.cuda().float())
        prediction1 = self.netG_A(self.real_A.cuda().float(),self.layers_last)
        prediction2 = self.netAdhoc(prediction1)
        label = self.input_label.long().cuda()

        #label = self.input_label.float().cuda()

        #print(prediction)
        #print(label)

        print("XXXXXXXXXXX")
        print(label)
        print(prediction2)
        loss_G = self.criterionCrossEntropy(prediction2, self.lab.long().cuda())
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


    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netAdhoc, 'Adhoc', label, self.gpu_ids)
        #self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)



