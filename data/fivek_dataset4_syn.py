import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform,no_transform, get_transform_vgg_A, get_transform_vgg_B, get_transform_vgg_C, get_transform_vgg_D, get_transform_vgg_E, get_transform_vgg_F, get_transform_vgg_G
from data.image_folder import make_dataset
from PIL import Image
import random

class FiveKDataset4_syn(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        if opt.phase == 'train':
            self.dir_C = os.path.join(opt.dataroot+'trainB_exC_resized')

        if opt.phase == 'test':
            self.dir_C = os.path.join(opt.dataroot+'testB_exC_resized')


        self.C_paths = make_dataset(self.dir_C)
        self.C_paths = sorted(self.C_paths)

        self.transform_A = get_transform_vgg_A(opt)
        self.transform_B = get_transform_vgg_B(opt)
        self.transform_C = get_transform_vgg_C(opt)
        self.transform_D = get_transform_vgg_D(opt)
        self.transform_E = get_transform_vgg_E(opt)
        self.transform_F = get_transform_vgg_F(opt)
        self.transform_G = get_transform_vgg_G(opt)


    def __getitem__(self, index):
        
        C_path = self.C_paths[index]

        #ran_num = (int)(random.random()*7)
        ran_num = 3

        C_img = Image.open(C_path).convert('RGB')

        if ran_num == 0:
            In = self.transform_A(C_img)
        if ran_num == 1:
            In = self.transform_B(C_img)
        if ran_num == 2:
            In = self.transform_C(C_img)
        if ran_num == 3:
            In = self.transform_D(C_img)
        if ran_num == 4:
            In = self.transform_E(C_img)
        if ran_num == 5:
            In = self.transform_F(C_img)
        if ran_num == 6:
            In = self.transform_G(C_img)
            
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'In': In,'Expert': ran_num}

    def __len__(self):
        return len(self.C_paths)

    def name(self):
        return 'FiveKDataset4_syn'
