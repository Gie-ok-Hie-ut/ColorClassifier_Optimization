#####
#
#             AVA Metric Learning 
######


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

class ColorClassify_Model3(BaseModel):
    def name(self):
        return 'ColorClassify_Model3'

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

        
        #self.netG_A.load_state_dict(torch.load('./vgg_original.pth')) #original vgg
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
          

            #self.netG_A.load_state_dict(torch.load('./15_net_G_A.pth')) #original vgg
            #self.netG_A.load_state_dict(torch.load('./vgg_original.pth')) #original vgg
            self.netG_A.load_state_dict(torch.load('./vgg_original.pth')) #original vgg
            self.load_network(self.netG_A, 'G_A', which_epoch)
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
                                                 #self.netAdhoc.parameters(),
                                                 ),lr=opt.lr, betas=(opt.beta1, 0.999))
            
            #self.optimizer_G = torch.optim.Adadelta(itertools.chain(
            #                                     self.netG_A.parameters(),
            #                                     ),lr=opt.lr, rho = 0.95, eps = 1e-06)




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
        self.input_A =  input['A']
        self.input_AL = input['AL']
        self.input_B =  input['B']
        self.input_BL = input['BL']
        self.input_C =  input['C']
        self.input_CL = input['CL']    


    def to_tensor_label(self,label):
        a = np.zeros(self.num_last_class)
        a[label]=1
        b= torch.from_numpy(a).unsqueeze(0)
        return b


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
        self.real_A  = Variable(self.input_A)
        self.real_AL = Variable(self.input_AL)
        self.real_B  = Variable(self.input_B)
        self.real_BL = Variable(self.input_BL)
        self.real_C  = Variable(self.input_C)
        self.real_CL = Variable(self.input_CL)        

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

    def L2dist(self,x1,x2,norm):

        #eps = 1e-4 / x1.size(0)
        eps = 1e-6

        #print(x1)
        #print(x1.type())
        #print(x1.size())
        #print(x2.size())


        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, norm).sum(dim=1)
        #print("x1")
        #print(x1)
        #print("diff")
        #print(diff)
        #print("out")
        #print(out)
        return torch.pow(out + eps, 1. / norm)
        #epsilon = 1e-6
        #return ((x1-x2).pow(2).sum(1)+epsilon).sqrt()


    def backward_all(self):
        #print(self.real_A.cuda().float().size())
        #print(self.real_B.cuda().float().size())
        #print(memorys)
        featureA = self.netG_A(self.real_A.cuda().float(), self.layers_last)
        featureB = self.netG_A(self.real_B.cuda().float(), self.layers_last)
        featureC = self.netG_A(self.real_C.cuda().float(), self.layers_last)

        labelA = self.real_AL[0][0].cuda().float()
        labelB = self.real_BL[0][0].cuda().float()
        labelC = self.real_CL[0][0].cuda().float()


        epsilon = 1e-6

        #print(self.real_A.cuda().float())
        #print(self.real_A.cuda().float().type())
        #print(self.real_A.cuda().float().size())
        #print(featureA[0])

        #print(np.asarray(featureA[0]))
        #print(np.asarray(featureA[0]).dtype)
        #print(torch.from_numpy(np.asarray(featureA[0])))
        #print(featureA.type())
        #print(featureA.size())
        #print(MANNAT)
        #print(labelA)
        #print(labelB)
        #print(labelC)

        if (labelA > labelB) and (labelA > labelC):
            #log_dist1 = torch.log(self.L2dist(featureA[0], featureB[0],2) + epsilon).mean()
            log_dist1 = torch.log(self.L2dist(featureA[0], featureB[0],2).mean() + epsilon)
            log_gt_dist1 = torch.log((labelA - labelB)*1 + epsilon)
            #log_dist2 = torch.log(self.L2dist(featureA[0], featureC[0],2) + epsilon).mean()
            log_dist2 = torch.log(self.L2dist(featureA[0], featureC[0],2).mean() + epsilon)
            log_gt_dist2 = torch.log((labelA - labelC)*1 + epsilon)        

            log_ratio_loss = ((log_dist1 - log_dist2) - (log_gt_dist1 - log_gt_dist2)).pow(2)



            loss = log_ratio_loss.sum()

            print("LOG")
            print(log_dist1.data)
            print(log_gt_dist1.data)
            print(log_dist2.data)
            print(log_gt_dist2.data)
            loss.backward()
            self.loss_G = loss.data
            print(self.loss_G)
            self.optimizer_G.step()
        
        #CDF
        # torch.cumsum // cumulative sum
        #else:
        #    loss = torch.tensor(0).float().cuda()
            #if(np.isnan(log_dist1.cpu().detach().numpy())):
            #    print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            #if(np.isnan(log_dist2.cpu().detach().numpy())):
            #    print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")

            #print("ROLLING")
            #print(log_dist2.cpu().detach().numpy())
            #if(log_dist2.cpu().detach().numpy()== -6.9067564):
            #    print(irrefer)





        #print(log_ratio_loss)



        #prediction2 = self.netAdhoc(prediction1)
        #label = self.input_label.long().cuda()

        #label = self.input_label.float().cuda()

        #print(prediction)
        #print(label)

        
        #print(label)
        #print(prediction2)
        #loss_G = self.criterionCrossEntropy(prediction2, self.lab.long().cuda())
        #loss_G.backward()

        #print(ji)  



        

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.forward()
        self.backward_all()
        #self.optimizer_G.step()

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




