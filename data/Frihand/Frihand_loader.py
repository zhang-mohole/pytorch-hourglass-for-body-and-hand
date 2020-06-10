import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import data.Frihand.fh_utils as fh_utils
# import fh_utils
import os

class Frihand_Dataset(Dataset):
    def __init__(self, img_root, uv_list, resize_img=True, hm_size=64, joints_num=21, period='train'):
        
        if period == 'train':
            ix_start = 0
            ix_end = 28000
        elif period == 'valid':
            ix_start = 28000
            ix_end = 30800
        elif period == 'test':
            ix_start == 30800
            ix_end == 32560
        else:
            assert 'invalid period setting, only [training, validation, test] available'
        set_name = 'training'

        self.img_li = [os.path.join(img_root, set_name, 'rgb','%08d.jpg' % i) for i in range(ix_start, ix_end)]
        self.uv_li = uv_list[ix_start:ix_end]
        assert len(self.img_li)==len(self.uv_li), 'image and uv list length not match'
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if resize_img:
            self.transform = transforms.Compose([transforms.Resize(256),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.resize_img = resize_img
        self.hm_size = hm_size
        self.hm_generater = GenerateHeatmap(hm_size, joints_num)
            

    def __getitem__(self, index):
        img = self.transform(Image.open(self.img_li[index]))
        ori_uv = np.array(self.uv_li[index])
        uv = ori_uv*self.hm_size/224
        uv = uv.astype(int)
        hm = self.hm_generater(uv)
        hm = torch.from_numpy(hm)
        return img, hm

    def __len__(self):
        return len(self.img_li)


class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(keypoints):
            if pt[0] > 0: 
                x, y = int(pt[0]), int(pt[1])
                if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms

def get_Frihand_loader(batch_size=8, period='train'):
    data_root = '/home/mhao/data/FreiHAND_pub_v2'
    uv_li = []
    K_list, _, xyz_list = fh_utils.load_db_annotation(data_root, set_name='training')
    for i, xyz in enumerate(xyz_list):
        uv = fh_utils.projectPoints(xyz, K_list[i])
        uv_li.append(uv)
    
    train_data = Frihand_Dataset(data_root, uv_li, period=period)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    return train_loader

if __name__ == "__main__":
    data_root = '/home/mhao/data/FreiHAND_pub_v2'
    uv_li = []
    K_list, _, xyz_list = fh_utils.load_db_annotation(data_root, set_name='training')
    for i, xyz in enumerate(xyz_list):
        uv = fh_utils.projectPoints(xyz, K_list[i])
        uv_li.append(uv)
    
    test_data = Frihand_Dataset(data_root, uv_li)
    test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=False)

    for img, hm in test_loader:
        print(img.shape)
        print(hm.shape)
        break

