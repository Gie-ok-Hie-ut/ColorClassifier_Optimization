import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform,no_transform, get_transform_vgg
from data.image_folder import make_dataset
from PIL import Image
import random

class FiveKDataset4(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        if opt.phase == 'train':
            self.dir_A = os.path.join(opt.dataroot+'trainB_exA_resized')
            self.dir_B = os.path.join(opt.dataroot+'trainB_exB_resized')
            self.dir_C = os.path.join(opt.dataroot+'trainB_exC_resized')
            self.dir_D = os.path.join(opt.dataroot+'trainB_exD_resized')
            self.dir_E = os.path.join(opt.dataroot+'trainB_exE_resized')
            self.dir_M = os.path.join(opt.dataroot+'trainB_exA_resized')  ##### M
            self.dir_N = os.path.join(opt.dataroot+'trainB_exB_resized')  ##### N
        #self.dir_A = os.path.join(opt.dataroot+'Jpeg/expertA_test')
        #self.dir_B = os.path.join(opt.dataroot+'Jpeg/expertB_test')
        #self.dir_C = os.path.join(opt.dataroot+'Jpeg/expertC_test')
        #self.dir_D = os.path.join(opt.dataroot+'Jpeg/expertD_test')
        #self.dir_E = os.path.join(opt.dataroot+'Jpeg/expertE_test')
            self.dir_R = os.path.join(opt.dataroot+'trainA')

        if opt.phase == 'test':
            self.dir_A = os.path.join(opt.dataroot+'testB_exA_resized')
            self.dir_B = os.path.join(opt.dataroot+'testB_exB_resized')
            self.dir_C = os.path.join(opt.dataroot+'testB_exC_resized')
            self.dir_D = os.path.join(opt.dataroot+'testB_exD_resized')
            self.dir_E = os.path.join(opt.dataroot+'testB_exE_resized')
            self.dir_M = os.path.join(opt.dataroot+'testB_exA_resized') ##### M
            self.dir_N = os.path.join(opt.dataroot+'testB_exB_resized') ##### N
            
        #self.dir_A = os.path.join(opt.dataroot+'Jpeg/expertA_test')
        #self.dir_B = os.path.join(opt.dataroot+'Jpeg/expertB_test')
        #self.dir_C = os.path.join(opt.dataroot+'Jpeg/expertC_test')
        #self.dir_D = os.path.join(opt.dataroot+'Jpeg/expertD_test')
        #self.dir_E = os.path.join(opt.dataroot+'Jpeg/expertE_test')
            self.dir_R = os.path.join(opt.dataroot+'testA')


        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.C_paths = make_dataset(self.dir_C)
        self.D_paths = make_dataset(self.dir_D)
        self.E_paths = make_dataset(self.dir_E)
        self.M_paths = make_dataset(self.dir_M)
        self.N_paths = make_dataset(self.dir_N)
        self.R_paths = make_dataset(self.dir_R)


        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.C_paths = sorted(self.C_paths)
        self.D_paths = sorted(self.D_paths)
        self.E_paths = sorted(self.E_paths)
        self.M_paths = sorted(self.M_paths)
        self.N_paths = sorted(self.N_paths)
        self.R_paths = sorted(self.R_paths)

        self.M_size = len(self.M_paths)
        self.N_size = len(self.N_paths)

        self.transform = get_transform_vgg(opt)


    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]
        D_path = self.D_paths[index]
        E_path = self.E_paths[index]
        M_path = self.M_paths[index % self.M_size] # There's not much
        N_path = self.N_paths[index % self.N_size]
        R_path = self.R_paths[index]

        ran_num = (int)(random.random()*4)
        ran_num2 = (int)(random.random()*4)


        if ran_num == 0:
            B_img = Image.open(B_path).convert('RGB')
            In = self.transform(B_img)
        if ran_num == 1:
            C_img = Image.open(C_path).convert('RGB')
            In = self.transform(C_img)
        if ran_num == 2:
            E_img = Image.open(E_path).convert('RGB')
            In = self.transform(E_img)
        if ran_num == 3:
            N_img = Image.open(N_path).convert('RGB') # RED
            In = self.transform(N_img)


        if ran_num2 == 0:
            B_img = Image.open(B_path).convert('RGB')
            In2 = self.transform(B_img)
        if ran_num2 == 1:
            C_img = Image.open(C_path).convert('RGB')
            In2 = self.transform(C_img)
        if ran_num2 == 2:
            E_img = Image.open(E_path).convert('RGB')
            In2 = self.transform(E_img)
        if ran_num2 == 3:
            N_img = Image.open(N_path).convert('RGB') # RED
            In2 = self.transform(N_img)


        #if ran_num == 0:
        #    B_img = Image.open(B_path).convert('RGB')
        #    In = self.transform(B_img)
        #if ran_num == 1:
        #    E_img = Image.open(E_path).convert('RGB')
        #    In = self.transform(E_img)
        
        #if ran_num== 0:
        #    In = self.transform(A_img)
        #if ran_num== 1:
        #    In = self.transform(B_img)

            
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'In': In,'In2':In2, 'Expert': ran_num, 'Expert2': ran_num2}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'FiveKDataset4'
