import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform,no_transform, get_transform_vgg
from data.image_folder import make_dataset
from PIL import Image
from PIL import ImageFile
import random
import numpy as np
import os

class AVADataset(BaseDataset):    
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        # Image - AVA
        self.dir_A = os.path.join(opt.dataroot+'image')
        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)

        # Label - AVA
        self.dir_BL = os.path.join(opt.dataroot+'split_new')
        self.BL_paths = make_dataset(self.dir_BL, mode= 'np')
        self.BL_paths = sorted(self.BL_paths)

        # Label - Anchor
        self.dir_AL = os.path.join(opt.dataroot+'split_above7')
        self.AL_paths = make_dataset(self.dir_AL, mode= 'np')
        self.AL_paths = sorted(self.AL_paths)


        # Size
        self.AVA_size = len(self.A_paths)
        self.Anchor_size= len(self.AL_paths)
        #self.AVA_size = self.AVA_size - 100


        #self.transform = get_transform_vgg(opt)
        self.transform = no_transform(opt)

        ImageFile.LOAD_TRUNCATED_IMAGES = True



    def __getitem__(self, index):

        test_data_num = 100 # Last 100 data would not be used

        # Name - Random Seed
        A_num = ((random.randint(0, self.Anchor_size -1 - test_data_num)) % self.Anchor_size)
        B_num = ((random.randint(0, self.AVA_size -1    - test_data_num)) % self.AVA_size)
        C_num = ((random.randint(0, self.AVA_size -1    - test_data_num)) % self.AVA_size)

        # Name - Label
        AL_path = self.AL_paths[A_num]
        BL_path = self.BL_paths[B_num]
        CL_path = self.BL_paths[C_num] # Same Directory
        #print(AL_path)
        #print(BL_path)
        #print(CL_path)

        # Name - Img
        A_name = os.path.splitext(os.path.basename(AL_path))[0]
        B_name = os.path.splitext(os.path.basename(BL_path))[0]
        C_name = os.path.splitext(os.path.basename(CL_path))[0]

        A_path = self.opt.dataroot + 'image/' + A_name + '.jpg'
        B_path = self.opt.dataroot + 'image/' + B_name + '.jpg'
        C_path = self.opt.dataroot + 'image/' + C_name + '.jpg'


        # Open - Label
        AL = np.load(AL_path)
        BL = np.load(BL_path)
        CL = np.load(CL_path)

        # Open - Img
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        B_img = Image.open(B_path).convert('RGB')
        B = self.transform(B_img)
        C_img = Image.open(C_path).convert('RGB')
        C = self.transform(C_img)        

        return {'A': A, 'AL': AL, 'B': B, 'BL': BL, 'C': C, 'CL': CL}


    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AVADataset'
