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

class ColorClassify_Model(BaseModel):
    def name(self):
        return 'ColorClassify_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize

        self.num_last_class= 3 # 0 ~ num_class-1
        self.interest_class = 0

        #self.netG_A = networks.define_G(3, 3,opt.ngf, 'resonly', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        #self.netG_A.cuda()
        #self.netG_A = networks.define_pretrained('vgg19') # 'vgg19'
        #self.netG_A = networks.define_F(self.gpu_ids,extract=True,use_input_norm=False,use_bn=False,colorcheckpoint=True)
        #self.netF_A = networks.define_F(self.gpu_ids,extract=True,use_input_norm=True,use_bn=False,colorcheckpoint=False)
        #self.netG_A = networks.define_pretrained('nasnetalarge') # 'vgg19'


        self.netG_A = networks.define_VGG()
        self.netG_A.cuda()
        print(self.netG_A)
        print(self.netG_A.state_dict().keys())        
        
        self.netAdhoc = networks.define_adhoc2(self.num_last_class)
        self.netAdhoc.cuda()


        # Layers
        self.layers_last = ['p5']
        self.layers_extract = ['r21','r31','r41', 'r51'] # Extract

        self.string_centroid_r21 = './centroid/Syn7_ep10/centroid_Syn_night_r21.pt'
        self.string_centroid_r31 = './centroid/Syn7_ep10/centroid_Syn_night_r31.pt'
        self.string_centroid_r41 = './centroid/Syn7_ep10/centroid_Syn_night_r41.pt'
        self.string_centroid_r51 = './centroid/Syn7_ep10/centroid_Syn_night_r51.pt'

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

            self.centroid1 = torch.zeros(1,512,28,28).cuda()
            self.centroid2 = torch.zeros(1,512,28,28).cuda()
            self.centroid3 = torch.zeros(1,512,28,28).cuda()
            self.centroid4 = torch.zeros(1,512,28,28).cuda()

            self.centroid_r21 = torch.zeros(1,128,112,112).cuda()
            self.centroid_r31 = torch.zeros(1,256,56,56).cuda()
            self.centroid_r41 = torch.zeros(1,512,28,28).cuda()
            self.centroid_r51 = torch.zeros(1,512,14,14).cuda()

            self.dis_all = torch.zeros(3,1)


        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
          
            
            #self.netG_A.load_state_dict(torch.load('./15_net_G_A.pth')) #original vgg
            #self.netG_A.load_state_dict(torch.load('./vgg_conv.pth')) #original vgg
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netAdhoc,'Adhoc',which_epoch)
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
        input_label = input['Expert']
      
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



    def extract(self):
        with torch.no_grad():
            real_A = Variable(self.input_A)
            lab = Variable(self.lab)
            real_A = real_A.float().cuda()

            prediction = self.netG_A(real_A,self.layers_last)
            self.statistics[lab.long().cuda()] = self.statistics[lab.long().cuda()] + 1

            
            if lab.long() == 0:        
            	self.centroid1 = self.centroid1 + prediction[0]
            if lab.long() == 1:
            	self.centroid2 = self.centroid2 + prediction[0]
            if lab.long() == 2:
            	self.centroid3 = self.centroid3 + prediction[0]
            if lab.long() == 3:
                self.centroid4 = self.centroid4 + prediction[0]

            print(self.statistics)

    def extract_multi(self):
        with torch.no_grad():
            real_A = Variable(self.input_A)
            lab = Variable(self.lab)
            real_A = real_A.float().cuda()

            prediction = self.netG_A(real_A,self.layers_extract)
            self.statistics[lab.long().cuda()] = self.statistics[lab.long().cuda()] + 1


            
            if lab.long() == self.interest_class:
                print("True")        
                self.centroid_r21 = self.centroid_r21 + prediction[0]
                self.centroid_r31 = self.centroid_r31 + prediction[1]
                self.centroid_r41 = self.centroid_r41 + prediction[2]
                self.centroid_r51 = self.centroid_r51 + prediction[3]
            else:
            	print("Falseeeeeeeeeeeeeeeee")

            print(self.statistics)

    def calc_centroid(self):

        self.centroid1 = self.centroid1/self.statistics[0]
        self.centroid2 = self.centroid2/self.statistics[1]
        self.centroid3 = self.centroid3/self.statistics[2]

        print(self.centroid1)
        print(self.centroid2)
        print(self.centroid3)
        print(self.centroid3)

    def calc_centroid_multi(self):
    	self.centroid_r21 = self.centroid_r21/(self.statistics[self.interest_class]-1)
    	self.centroid_r31 = self.centroid_r31/(self.statistics[self.interest_class]-1)
    	self.centroid_r41 = self.centroid_r41/(self.statistics[self.interest_class]-1)
    	self.centroid_r51 = self.centroid_r51/(self.statistics[self.interest_class]-1)

    def save_centroid(self):
        torch.save(self.centroid1,'./centroid1.pt')
        torch.save(self.centroid2,'./centroid2.pt')
        torch.save(self.centroid3,'./centroid3.pt')

    def save_centroid_multi(self):
        torch.save(self.centroid_r21,self.string_centroid_r21)
        torch.save(self.centroid_r31,self.string_centroid_r31)
        torch.save(self.centroid_r41,self.string_centroid_r41)
        torch.save(self.centroid_r51,self.string_centroid_r51)

    def load_centroid(self):
        self.centroid1=torch.load('./centroid1.pt')
        self.centroid2=torch.load('./centroid2.pt')
        self.centroid3=torch.load('./centroid3.pt')

    def calc_transfer(self):
        real_A = Variable(self.input_A,requires_grad = False)
        out_img = Variable(self.input_A.data.clone(),requires_grad = True)

        optimizer = torch.optim.LBFGS([out_img])
        criterionMSE = torch.nn.MSELoss()

        run = [0]
        while run[0] <= 50:
            def closure():
                optimizer.zero_grad()
                prediction = self.netG_A(out_img)

                prediction2 = self.netF_A(out_img)
                prediction3 = self.netF_A(real_A)

                loss1 = criterionMSE(prediction,self.centroid2)
                loss2 = criterionMSE(prediction2,prediction3)
                loss = 0.3*loss1 +loss2
                loss.backward()
                return loss
            optimizer.step(closure)
            run[0] += 1
            print(".")
        print("FINISHED")

        self.real_A = real_A.data
        self.fake_B = out_img.data





    def calc_distance(self):
        real_A = Variable(self.input_A)
        lab = Variable(self.lab)

        real_A = real_A.float().cuda()

        prediction = self.netG_A(real_A)


        self.dis = torch.zeros(3,1)
        self.dis[0] = torch.sqrt(torch.sum((prediction - self.centroid1) * (prediction - self.centroid1)))
        self.dis[1] = torch.sqrt(torch.sum((prediction - self.centroid2) * (prediction - self.centroid2)))
        self.dis[2] = torch.sqrt(torch.sum((prediction - self.centroid3) * (prediction - self.centroid3)))

        #print(self.centroid1) #1/512/14/14
        #print(prediction.permute(1,0,2,3)) #512/1/14/14
        #print(prediction) #512/1/14/14
        #print("YAYA")
        
        #self.dis[0] = torch.sqrt(torch.mm(self.centroid1,prediction))
        #self.dis[1] = torch.sqrt(torch.mm(self.centroid2,prediction))
        #self.dis[2] = torch.sqrt(torch.mm(self.centroid3,prediction))
        


        self.dis_all = torch.cat((self.dis_all,self.dis),1)



        value, indices = torch.min(self.dis,0)

        #print(lab.long().cuda())
        #print(indices)
        self.statistics_cent[lab.long().cuda()] = self.statistics_cent[lab.long().cuda()] + 1

        if indices == lab.long() :
            self.correct_cent = self.correct_cent + 1
            self.statistics_acc_cent[indices] = self.statistics_acc_cent[indices] + 1
        else:
            self.wrong_cent = self.wrong_cent + 1

        ratio = self.correct_cent / (self.correct_cent + self.wrong_cent)

        
        print(ratio)
        print(self.statistics_acc_cent)
        print(self.statistics_cent)
        print(self.statistics_acc_cent/self.statistics_cent)
        print(torch.mean(self.dis_all,1))
        print(torch.std(self.dis_all,1))


        
        


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

    def optimize_parameters_soft(self):
        self.optimizer_G.zero_grad()
        self.forward()
        self.backward_all_soft()
        self.optimizer_G.step()


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

    def get_current_visuals_transfer(self,num):
        real_A = util.tensor2im(self.real_A,False)
        fake_B = util.tensor2im(self.fake_B,False)
        #fake_B = util.tensor2im(self.fake_B, False)
        
        ret_visuals = OrderedDict([('real_A'+str(num), real_A), 
                                   ('fake_B'+str(num), fake_B),
         ])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netAdhoc, 'Adhoc', label, self.gpu_ids)
        #self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)



