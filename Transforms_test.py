from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "dataset/train/bees_image/39747887_42df2855ee.jpg"
img = Image.open(img_path)  # 得到一个PIL的Image
img_transforms = transforms.ToTensor()  # 实例化一个Totensor对象
img_tensor = img_transforms(img)  # 得到图像经过totensor的数据,得到tensor数据类型的照片数据
writer = SummaryWriter("logs")
writer.add_image("Tensor_img", img_tensor, 1)
writer.close()
