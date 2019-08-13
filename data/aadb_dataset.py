import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform,no_transform, get_transform_vgg
from data.image_folder import make_dataset
from PIL import Image
import random

class AADBDataset(BaseDataset):    
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        if opt.phase == 'train':
            self.dir_A = os.path.join(opt.dataroot+'trainB_exA_resized')
            self.dir_B = os.path.join(opt.dataroot+'trainB_exB_resized')

        if opt.phase == 'test':
            self.dir_A = os.path.join(opt.dataroot+'testB_exA_resized')
            self.dir_B = os.path.join(opt.dataroot+'testB_exB_resized')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)

        self.transform = get_transform_vgg(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]


        ran_num = (int)(random.random()*2)

        if ran_num == 0:
            A_img = Image.open(A_path).convert('RGB')
            In = self.transform(A_img)
        if ran_num == 1:
            B_img = Image.open(B_path).convert('RGB')
            In = self.transform(B_img)


            
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'In': In, 'Label': ran_num}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AADBDataset'
