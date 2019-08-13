import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform,no_transform, get_transform_vgg
from data.image_folder import make_dataset
from PIL import Image
import random

class FiveKDataset_single(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        if opt.phase == 'test':
            self.dir_Single = os.path.join(opt.dataroot+'testB_singledata')

        self.Single_paths = make_dataset(self.dir_Single)
        self.Single_paths =sorted(self.Single_paths)
        self.Single_size = len(self.Single_paths)

        self.transform = get_transform_vgg(opt)


    def __getitem__(self, index):
        Single_path = self.Single_paths[index % self.Single_size]

        Single_img = Image.open(Single_path).convert('RGB') # RED
        In = self.transform(Single_img)
        In2 = In
        ran_num=0
        ran_num2 = 0

            
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'In': In,'In2':In2, 'Expert': ran_num, 'Expert2': ran_num2}

    def __len__(self):
        return len(self.Single_paths)

    def name(self):
        return 'FiveKDataset_single'
