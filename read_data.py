from torch.utils.data import Dataset
import cv2
from PIL import Image
import os

class Mydata(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        # 根目录地址
        self.label_dir=label_dir
        # 标签
        self.path=os.path.join(self.root_dir,self.label_dir)
        # 图片所在的目录地址
        self.image_path=os.listdir(self.path)
        # 所有图片地址组成的列表


    def __getitem__(self, index):
        image_name=self.image_path[index]
        # 图片的名称
        image_item=os.path.join(self.root_dir,self.label_dir,image_name)
        # 图片的地址
        image=Image.open(image_item)
        # 打开一个图片,获取信息
        label=self.label_dir
        return image,label

    def __len__(self):
        return len(self.image_path)



root_dir="dataset/train"
label_dir="ants_image"
ants_dataset=Mydata(root_dir,label_dir)

root_dir="dataset/train"
target_dir="ants_image"
image_path=os.listdir(os.path.join(root_dir,target_dir))
label=target_dir.split('_')[0]
out_dir="ants_label"
for i in image_path:
    file_name=i.split(".jpg")[0]
    with open(os.path.join(root_dir,out_dir,"{}.txt".format(file_name)),'w') as f:
        f.write(label)

root_dir="dataset/train"
target_dir="bees_image"
image_path=os.listdir(os.path.join(root_dir,target_dir))
label=target_dir.split('_')[0]
out_dir="bees_label"
for i in image_path:
    file_name=i.split(".jpg")[0]
    with open(os.path.join(root_dir,out_dir,"{}.txt".format(file_name)),'w') as f:
        f.write(label)