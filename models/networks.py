import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
#from modules.architecture import VGGFeatureExtractor
import models.modules.architecture as arch
import torchvision
import torch.nn.functional as F
#from .architecture import VGGFeatureExtractor
###############################################################################
# Functions
###############################################################################
import pretrainedmodels
from torch.utils.model_zoo import load_url
import torch.utils.model_zoo as model_zoo


def define_pretrained(model_name):
    model = pretrainedmodels.__dict__[model_name](num_classes=1000,pretrained='imagenet')
    return model

def define_pretrained_vgg():
    model = torchvision.models.vgg19(pretrained=True)
    return model


def define_adhoc(int_num,out_num,gpu_ids=[]):
    netAdhoc = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())

    netAdhoc = Adhoc(int_num,out_num)

    if len(gpu_ids) > 0:
        netAdhoc.cuda(gpu_ids[0])
    init_weights(netAdhoc, init_type='normal')
    return netAdhoc

def define_adhoc2(out_num,gpu_ids=[]):
    netAdhoc = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())

    netAdhoc = VGG_Linear(out_num)

    if len(gpu_ids) > 0:
        netAdhoc.cuda(gpu_ids[0])
    #init_weights(netAdhoc, init_type='normal')
    return netAdhoc

def define_F(gpu_ids, colorcheckpoint= False, extract=False, use_input_norm=False, use_bn=False):
    tensor = torch.cuda.FloatTensor if gpu_ids else torch.FloatTensor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    #netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, use_input_norm=use_input_norm,colorcheckpoint=colorcheckpoint, extract= extract,tensor=tensor)
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, use_input_norm=use_input_norm,colorcheckpoint=colorcheckpoint, extract= extract,tensor=tensor)

    if gpu_ids:
        netF = nn.DataParallel(netF).cuda()
    netF.eval()  # No need to train
    return netF


class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=False,
                 colorcheckpoint=False,
                 extract = False,
                 tensor=torch.FloatTensor):
        super(VGGFeatureExtractor, self).__init__()
        

        if colorcheckpoint: #Pretrained 
            self.model = VGG19_Key(1000)
            save_path = './checkpoints/colorclassify_pretrained_vgg19_BCE/10_net_G_A.pth'        
            print(self.model)
            print(self.model.state_dict().keys())
            self.model.load_state_dict(torch.load(save_path))
        else: # VGG 19
            self.model = torchvision.models.vgg19(pretrained=True)
            

        #if use_bn:
#
#        #    self.model = VGG19_Key(1000)
#        #    #model = torchvision.models.vgg19_bn(pretrained=True)
#        #    #model=pretrainedmodels.__dict__[model_name](num_classes=1000,pretrained='imagenet')
#        #else:
#        #    self.model = VGG19_Key(1000)
#        #    #model = torchvision.models.vgg19(pretrained=True)
#        #    #model=pretrainedmodels.__dict__['vgg19'](num_classes=1000,pretrained='imagenet')
        ##model=pretrainedmodels.__dict__['vgg19'](num_classes=1000,pretrained='imagenet')
        
        #self.model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
        
        


        # When
        self.extract = extract
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = Variable(tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = Variable(tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        #print("SALM JJALBGO")
        #print(model)
        #print(type(model)) # <class 'torchvision.models.vgg.VGG'>
        #print(model.features)
        #print(model.features(0))        
        #print(model.locals)
        #print("SARANGH")
        #print(model.features[1])
        #print(XXX)

        # stop at feature_layer+1


        if self.extract == True:
            if colorcheckpoint:
                self.features = nn.Sequential(*list(self.model._features.children())[:(feature_layer + 1)]) #Initial value
            else:
                self.features = nn.Sequential(*list(self.model.features.children())[:(feature_layer + 1)]) #Initial value
            # No need to BP to variable
            for k, v in self.features.named_parameters():
                v.requires_grad = False        

        #print(self.features)
        #self.features = nn.Sequential(*list(model.children())[:(feature_layer + 1)])
        #self.features2 = nn.Sequential(*list(model.children())[:(10)])

        #print("PRIMETIME")
        #print(self.features)
        #print("MOFUCKIN")
        #print(self.features2)
        #print(gkgkgk)



    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        if self.extract:
            output = self.features(x)
        else:
            output = self.model(x)

        return output


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self._features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),
        )

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        return x



def define_VGG():
    netF = VGG()
    netF.eval()  # No need to train
    return netF

class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x, out_keys):

        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

class VGG_Linear(nn.Module):
    def __init__(self,out_num):
        super(VGG_Linear, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, out_num)
        )

    def forward(self,x):        
        x = x[0].view(x[0].size(0), 25088)
        x = self.classifier(x)
        return x


class VGG_Linear_multiple(nn.Module):
    def __init__(self,out_num):
        super(VGG_Linear, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, out_num)
        )

    def forward(self,x,y):        
        x = x[0].view(x[0].size(0), 25088)
        x = self.classifier(x)
        return x


class VGG19_Key(nn.Module):
    def __init__(self, num_classes):
        super(VGG19_Key, self).__init__()
        self._features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1, ceil_mode=False),
        )
        
        self.linear0 = nn.Linear(25088, 4096)
        self.linear0_aft = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            )

        self.linear1 = nn.Linear(4096, 4096)
        self.linear1_aft = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(),
            )
        self.last_linear = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), 25088)
        x = self.linear0(x)
        x = self.linear0_aft(x)
        x = self.linear1(x)
        x = self.linear1_aft(x)
        x = self.last_linear(x)
        return x

class Adhoc(nn.Module):
    def __init__(self,in_num,out_num):
        super(Adhoc,self).__init__()
        model = [nn.Linear(in_num,out_num),nn.Softmax()]
        self.model = nn.Sequential(*model)
    def forward(self,input):
        return self.model(input)


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='none', use_dropout=False, init_type='normal', gpu_ids=[]): # Batch norm -> nonw
    netG = None
    use_gpu = len(gpu_ids) > 0

    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_3blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
    elif which_model_netG == 'resonly':
        init_type='xavier'
        netG = RESONLY(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)


    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])

    init_weights(netD, init_type=init_type)
    return netD

#def define_F(gpu_ids, use_bn=False):
#    tensor = torch.cuda.FloatTensor if gpu_ids else torch.FloatTensor
#    # pytorch pretrained VGG19-54, before ReLU.
#    if use_bn:
#        feature_layer = 49
#    else:
#        feature_layer = 34
#    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True, tensor=tensor)
#    if gpu_ids:
#        netF = nn.DataParallel(netF).cuda()
#    netF.eval()  # No need to train
#    return netF

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

#class LogRatioLoss(Function):
#    """Log ratio loss function. """
#    def __init__(self, lamb, alpha):
#        super(LogRatioLoss, self).__init__()
#        self.mnd = 10
#        self.mxd = 100
#        self.pdist = L2dist(2)  # norm 2
#        self.lamb = lamb
#        self.alpha = alpha
#
#    def forward(self, input, gt_dist):
#        m = input.size()[0]-1   # #paired
#        a = input[0]            # anchor
#        p = input[1:]           # paired
#        
#        #  auxiliary variables
#        idxs = torch.range(1, m).cuda()
#        indc = idxs.repeat(m,1).t() < idxs.repeat(m,1)
#
#        epsilon = 1e-6
#
#        dist = self.pdist.forward(a,p)
#        # dist = ((a-p).pow(2).sum(1)+epsilon).sqrt()
#        # gt_dist[gt_dist>self.mxd] = self.mxd
#
#        log_dist = torch.log(dist + epsilon)
#        log_gt_dist = torch.log(gt_dist + epsilon)
#        diff_log_dist = log_dist.repeat(m,1).t()-log_dist.repeat(m, 1)
#        diff_log_gt_dist = log_gt_dist.repeat(m,1).t()-log_gt_dist.repeat(m, 1)
#
#        # uniform weight coefficients 
#        wgt = indc.clone().float()
#        wgt = wgt.div(wgt.sum())
#
#        log_ratio_loss = (diff_log_dist-diff_log_gt_dist).pow(2)
#
#        regularizer = (dist.repeat(m,1).t() + dist.repeat(m,1)).sub(self.alpha)
#        # regularizer = F.relu(regularizer)
#        regularizer = regularizer.clamp(min=1e-12)
#
#        loss = log_ratio_loss + self.lamb*regularizer
#        loss = loss.mul(wgt).sum()
#
#        return loss

#class RESONLY(nn.Module):
#    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='zero'):
#        super(RESONLY, self).__init__()
#
#        if type(norm_layer) == functools.partial:
#            use_bias = norm_layer.func == nn.InstanceNorm2d
#        else:
#            use_bias = norm_layer == nn.InstanceNorm2d
#
#        self.block_init = self.build_init_block(input_nc,ngf,padding_type, use_bias)
#        self.block_last = self.build_last_block(ngf,output_nc,padding_type,use_bias)
#
#        self.RES_Block1 = ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias)
#        self.RES_Block2 = ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias)
#        self.RES_Block3 = ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias)
#        self.RES_Block4 = ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias)
#        self.RES_Block5 = ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias)
#        self.RES_Block6 = ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias)
#        self.RES_Block7 = ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias)
#        self.RES_Block8 = ResnetBlock(ngf, padding_type, norm_layer, use_dropout, use_bias)
#
#        
#
# 
#    def build_init_block(self, input_nc,dim_img, padding_type, use_bias): # 3 -> 64
#        block_init =[]
#
#        p = 0
#        if padding_type == 'reflect':
#            block_init += [nn.ReflectionPad2d(1)]
#        elif padding_type == 'replicate':
#            block_init += [nn.ReplicationPad2d(1)]
#        elif padding_type == 'zero':
#            p = 1
#        else:
#            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#        
#        block_init += [nn.Conv2d(input_nc,dim_img,kernel_size=3,padding=p,stride=1)]
#        
#        return nn.Sequential(*block_init)
#
#
#    def build_last_block(self,dim_img,output_nc,padding_type,use_bias):
#        block_last = []
#
#        p = 0
#        if padding_type == 'reflect':
#            block_last += [nn.ReflectionPad2d(1)]
#        elif padding_type == 'replicate':
#            block_last += [nn.ReplicationPad2d(1)]
#        elif padding_type == 'zero':
#            p = 1
#        else:
#            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#
#        block_last += [nn.Conv2d(dim_img,dim_img,kernel_size=3,padding=p,bias=use_bias),
#                        nn.ReLU(True),
#                        nn.Conv2d(dim_img,dim_img,kernel_size=3,padding=p,bias=use_bias),
#                        nn.ReLU(True),
#                        nn.Conv2d(dim_img,dim_img,kernel_size=3,padding=p,bias=use_bias),
#                        nn.ReLU(True),
#                        nn.Conv2d(dim_img,output_nc,kernel_size=3,padding=p,bias=use_bias),
#                        nn.Tanh()
#                        ]
#
#        return nn.Sequential(*block_last)
#
#
#    def forward(self, input_img):
#        init_img = self.block_init(input_img) 
#
#        mid_img1 = self.RES_Block1.forward(init_img)
#        mid_img2 = self.RES_Block2.forward(mid_img1)
#        mid_img3 = self.RES_Block3.forward(mid_img2)
#        mid_img4 = self.RES_Block4.forward(mid_img3)
#        mid_img5 = self.RES_Block5.forward(mid_img4)
#        mid_img6 = self.RES_Block6.forward(mid_img5)
#        mid_img7 = self.RES_Block7.forward(mid_img6)
#        mid_img8 = self.RES_Block8.forward(mid_img7)
#
#        last_img = self.block_last(mid_img8)
#
#        return last_img

#class RESONLY(nn.Module):
#    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
#        assert(n_blocks >= 0)
#        super(RESONLY, self).__init__()
#        self.input_nc = input_nc
#        self.output_nc = output_nc
#        self.ngf = ngf
#        self.gpu_ids = gpu_ids
#        if type(norm_layer) == functools.partial:
#            use_bias = norm_layer.func == nn.InstanceNorm2d
#        else:
#            use_bias = norm_layer == nn.InstanceNorm2d
#
#        model = [nn.ReflectionPad2d(3),
#                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
#                           bias=use_bias),
#                 norm_layer(ngf),
#                 nn.ReLU(True)]
#
#        n_downsampling = 2
#        for i in range(n_downsampling):
#            mult = 2**i
#            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
#                                stride=2, padding=1, bias=use_bias),
#                      norm_layer(ngf * mult * 2),
#                      nn.ReLU(True)]
#
#        mult = 2**n_downsampling
#        for i in range(n_blocks):
#            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
#
#        for i in range(n_downsampling):
#            mult = 2**(n_downsampling - i)
#            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                         kernel_size=3, stride=2,
#                                         padding=1, output_padding=1,
#                                         bias=use_bias),
#                      norm_layer(int(ngf * mult / 2)),
#                      nn.ReLU(True)]
#        model += [nn.ReflectionPad2d(3)]
#        model += [nn.Conv2d(ngf, ngf*2, kernel_size=7, padding=0)]
#        model += [nn.AvgPool2d(4)]
#
#        model += [nn.ReflectionPad2d(3)]
#        model += [nn.Conv2d(ngf*2, ngf*4, kernel_size=7, padding=0)]
#        model += [nn.AvgPool2d(2)] #2048*32
#
#        model += [nn.ReflectionPad2d(3)]
#        model += [nn.Conv2d(ngf*4, ngf*4, kernel_size=7, padding=0)]
#        model += [nn.AvgPool2d(2)] #
#
#        model += [nn.ReflectionPad2d(1)]
#        model += [nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=0)]
#
#        model += [nn.ReflectionPad2d(1)]
#        model += [nn.Conv2d(ngf*8, ngf*16, kernel_size=4, stride=2, padding=0)]
#
#        model += [nn.ReflectionPad2d(1)]
#        model += [nn.Conv2d(ngf*16, ngf*32, kernel_size=4, stride=2, padding=0)]
#
#        model += [nn.ReflectionPad2d(1)]
#        model += [nn.Conv2d(ngf*32, ngf*64, kernel_size=4, stride=2, padding=0)]
#
#        #model += [nn.Linear(4096, 1000)] # 1 256 16 1000
#        #model += [nn.Linear(1000, 5)]
#        #model += [nn.Softmax()]
#
#        self.model = nn.Sequential(*model)
#
#    def forward(self, input):
#        print(input)
#        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#        else:
#            print(self.model(input))
#            return self.model(input)

class RESONLY(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=12, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(RESONLY, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, ngf*2, kernel_size=7, padding=0)]
        model += [nn.AvgPool2d(4)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf*2, ngf*4, kernel_size=7, padding=0)]
        model += [nn.AvgPool2d(4)] #2048*32

        #model += [nn.ReflectionPad2d(3)]
        #model += [nn.Conv2d(ngf*4, ngf*4, kernel_size=7, padding=0)]
        #model += [nn.AvgPool2d(2)] #
#
#        #model += [nn.ReflectionPad2d(1)]
#        #model += [nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=0)]
#
#        #model += [nn.ReflectionPad2d(1)]
#        #model += [nn.Conv2d(ngf*8, ngf*16, kernel_size=4, stride=2, padding=0)]
#
#        #model += [nn.ReflectionPad2d(1)]
#        #model += [nn.Conv2d(ngf*16, ngf*32, kernel_size=4, stride=2, padding=0)]
#
#        #model += [nn.ReflectionPad2d(1)]
        #model += [nn.Conv2d(ngf*32, ngf*64, kernel_size=4, stride=2, padding=0)]

        model2 = [nn.Linear(65536, 5000),nn.Dropout(p=0.5),nn.Linear(5000,1000), nn.Dropout(p=0.3), nn.Linear(1000, 5),] # 1 256 16 1000
        model3 = [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)

    def forward(self, input):
        a = self.model(input)
        a2 = a.view(a.size(0),-1)
        a3 = self.model2(a2)
        a4 = self.model3(a3)

        #print(a3)
        #print(a4)

        return a4

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

