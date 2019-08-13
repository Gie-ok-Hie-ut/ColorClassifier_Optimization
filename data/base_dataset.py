import torch.utils.data as data
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy
from skimage import color

#Added
import time
from math import exp
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def no_transform(opt):
    transform_list = []

    osize = [opt.loadSize, opt.loadSize]
    transform_list.append(transforms.Scale(osize, Image.BICUBIC))
    transform_list.append(transforms.RandomCrop(opt.fineSize))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_transform_lab(opt):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: RGB2LAB(numpy.array(img))))
    #transform_list.append(transforms.Lambda(lambda img: LAB2RGB(numpy.array(img))))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_transform_hsv(opt):
    transform_list = []

    transform_list.append(transforms.Lambda(lambda img: RGB2HSV(numpy.array(img))))
    transform_list.append(transforms.Lambda(lambda img: (numpy.array(img))))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_saturation(opt):
    transform_list = []



    #transform_list.append(transforms.Resize(interpolation=0.5))
    #osize = [opt.loadSize, opt.loadSize]
    #transform_list.append(transforms.Scale(osize, Image.BICUBIC))
    #transform_list.append(transforms.CenterCrop(opt.fineSize))
    #transform_list.append(transforms.RandomCrop(opt.fineSize))







    start_time = time.time()
    transform_list.append(transforms.Lambda(lambda img: RGB2HSV2RGB(numpy.array(img),"random",start_time)))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_grayish(opt):
    transform_list = []

    start_time = time.time()
    transform_list.append(transforms.Lambda(lambda img: RGB2HSV2RGB_Gray(numpy.array(img),"random",start_time)))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
#def get_transform_lab2(opt):
#    transform_list = []
#
#    transform_list.append(transforms.Lambda(lambda img: RGB2HSV(numpy.array(img))))
#    transform_list.append(transforms.Lambda(lambda img: (numpy.array(img))))
#
#    transform_list += [transforms.ToTensor(),
#                       transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
#                                            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))]
#    return transforms.Compose(transform_list)

def get_transform_vgg(opt):
    transform_list = []

    osize = [opt.loadSize, opt.loadSize]
    transform_list.append(transforms.Scale(osize, Image.BICUBIC))
    transform_list.append(transforms.RandomCrop(opt.fineSize))

    transform_list += [#transforms.Resize(opt.fineSize),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),  #subtract imagenet mean
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ]
    return transforms.Compose(transform_list)

def get_transform_vgg_A(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),1,1,1,1)))
    transform_list += [
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                           #                 (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),  #subtract imagenet mean
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ]
    return transforms.Compose(transform_list)

def get_transform_vgg_B(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),0.8,1.3,1,1)))
    transform_list += [
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                           #                 (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),  #subtract imagenet mean
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ]
    return transforms.Compose(transform_list)

def get_transform_vgg_C(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),0.8,1,1.3,1)))
    transform_list += [
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                           #                 (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),  #subtract imagenet mean
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ]
    return transforms.Compose(transform_list)

def get_transform_vgg_D(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),0.8,1,1,1.3)))
    transform_list += [
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                           #                 (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),  #subtract imagenet mean
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ]
    return transforms.Compose(transform_list)

def get_transform_vgg_E(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),0.8,1.4,1.4,1)))
    transform_list += [
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                           #                 (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),  #subtract imagenet mean
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ]
    return transforms.Compose(transform_list)

def get_transform_vgg_F(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),0.8,1.4,1,1.4)))
    transform_list += [
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                           #                 (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),  #subtract imagenet mean
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ]
    return transforms.Compose(transform_list)

def get_transform_vgg_G(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),0.8,1,1.4,1.4)))
    transform_list += [
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                           #                 (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),  #subtract imagenet mean
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ]
    return transforms.Compose(transform_list)


def get_transform_vgg_H(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),0.5,1,1,1)))
    transform_list += [
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                           #                 (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),  #subtract imagenet mean
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ]
    return transforms.Compose(transform_list)

def get_transform_vgg_H(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),0.0,1,1,1)))
    transform_list += [
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                           #                 (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],std=[1,1,1]),  #subtract imagenet mean
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ]
    return transforms.Compose(transform_list)


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)

def RGB2LAB(I):
    # AB 98.2330538631 -86.1830297444 94.4781222765 -107.857300207
    lab = color.rgb2lab(I)
    l = (lab[:, :, 0] / 100.0) * 255.0    # L component ranges from 0 to 100
    a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) * 255.0         # a component ranges from -127 to 127
    b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) * 255.0         # b component ranges from -127 to 127
    #l = (lab[:, :, 0] / 100.0) * 255.0    # L component ranges from 0 to 100
    #a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) * 255.0         # a component ranges from -127 to 127
    #b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) * 255.0         # b component ranges from -127 to 127
    return numpy.dstack([l, a, b])

def LAB2RGB(I):
    # print(I)
    l = I[:, :, 0] / 255.0 * 100.0
    a = I[:, :, 1] / 255.0 * (98.2330538631 + 86.1830297444) - 86.1830297444
    b = I[:, :, 2] / 255.0 * (94.4781222765 + 107.857300207) - 107.857300207
    # print(np.dstack([l, a, b]))

    rgb = color.lab2rgb(numpy.dstack([l, a, b]).astype(numpy.float64))
    return rgb

def RGB2HSV(I):
    hsv = color.rgb2hsv(I)
    h = (hsv[:, :, 0] / 360.0 ) * 255.0
    s = (hsv[:, :, 1] / 100.0 ) * 255.0
    v = (hsv[:, :, 2] / 100.0 ) * 255.0
    return numpy.dstack([h, s, v])

def Filter_syn(I,w,x,y,z):
    hsv = color.rgb2hsv(I)
    h = (hsv[:, :, 0] / 360.0 ) * 255.0
    s = (hsv[:, :, 1] / 100.0 ) * 255.0 * (w)
    v = (hsv[:, :, 2] / 100.0 ) * 255.0
    
    r = (h / 255.0 ) * 360.0
    g = (s / 255.0 ) * 100.0
    b = (v / 255.0 ) * 100.0


    rgb = color.hsv2rgb(numpy.dstack([r, g, b]).astype(numpy.float64))

    rgb[:,:,0] = rgb[:,:,0] * x
    rgb[:,:,1] = rgb[:,:,1] * y
    rgb[:,:,2] = rgb[:,:,2] * z


    rgb[:,:,0] = numpy.clip(rgb[:,:,0],0,1)
    rgb[:,:,1] = numpy.clip(rgb[:,:,1],0,1)
    rgb[:,:,2] = numpy.clip(rgb[:,:,2],0,1)


    return rgb


def RGB2HSV2RGB(I,mode,start_time):

    this_time=time.time()
    elapsed_time=this_time-start_time
    #print(this_time-start_time)


    if mode == "original":
        alpha = 1
    elif mode == "random":
        #adhoc = (elapsed_time*10000)%10
        #if adhoc >10:
        #    alpha = 0.1
        #else:
        #    alpha = ((elapsed_time*1000)%1000)*2/1000
        
        #alpha = ((elapsed_time*1000)%1000)*1/1000 # below 1


        ######ORIGINAL
        #alpha = ((elapsed_time*1000)%1000)*1.3/1000 # above 1

        #if alpha<0.1:
        #    alpha=0.1

        #if alpha>1.0: # to sustain identity loss
        #    alpha=1

        ###### Gray Added
        alpha = ((elapsed_time*1000)%1000)*(1.5)/1000 # above 1

        #if alpha>1.0 and alpha < 2.0:
        #    alpha=1.0
        #if alpha>=2.0:
        #    alpha=0.0

        #if alpha>1.0:
        #    alpha = 1.0
        #if alpha<0.5:
        #    alpha = 0.4

        if alpha > 1.0:
            alpha = 1.0
        if alpha <0.2:
            alpha = 0.2






        #if alpha>0.99  and alpha<1.4:
        #    alpha=0.1        

        #if alpha>0.99  and alpha<1.4:
        #    alpha=0.1

        #if alpha<3.0:
        #    alpha=3.0



    elif mode == "gray":
        alpha = 0.1
    elif mode == "decay":
        alpha = 1/(1+exp(0.0003*(elapsed_time-20000)))*0.9 + 0.1
    else:
        alpha = 1

    print("alpha:"+mode+":"+str(alpha))

    hsv = color.rgb2hsv(I)
    h = (hsv[:, :, 0] / 360.0 ) * 255.0
    s = (hsv[:, :, 1] / 100.0 ) * 255.0 * alpha
    v = (hsv[:, :, 2] / 100.0 ) * 255.0
    
    r = (h / 255.0 ) * 360.0
    g = (s / 255.0 ) * 100.0
    b = (v / 255.0 ) * 100.0


    rgb = color.hsv2rgb(numpy.dstack([r, g, b]).astype(numpy.float64))

    return rgb



def RGB2HSV2RGB_Gray(I,mode,start_time):


    hsv = color.rgb2hsv(I)
    h = (hsv[:, :, 0] / 360.0 ) * 255.0
    s = (hsv[:, :, 1] / 100.0 ) * 255.0 * 0.4
    v = (hsv[:, :, 2] / 100.0 ) * 255.0
    
    r = (h / 255.0 ) * 360.0
    g = (s / 255.0 ) * 100.0
    b = (v / 255.0 ) * 100.0


    rgb = color.hsv2rgb(numpy.dstack([r, g, b]).astype(numpy.float64))

    return rgb