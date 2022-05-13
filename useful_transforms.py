from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

image_path = "Images/wallhaven.jpg"

writer = SummaryWriter("logs")
image = Image.open(image_path)  # PIl数据类型
print(image)

# Totensor
trans_totensor = transforms.ToTensor()  # 实例化Totensor
tensor_image = trans_totensor(image)  # 将PIL数据类型的照片装换成tensor类型
writer.add_image("Totensor", tensor_image)

# Nomalize

trans_nomalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 实例化Nomalize，初始化的参数就是平均值和标准差
nomal_image = trans_nomalize(tensor_image)
writer.add_image("Nomalize", nomal_image, 1)
print(tensor_image[0][0][0])  # 比较标准化之后的数据区别
print(nomal_image[0][0][0])

# Resize
trans_Resize = transforms.Resize(512)  # 实例化一个Resize，参数是照片的大小，如果resize(512)表示将最短边调整至512，长宽比不变
# resize_image = trans_Resize(image)
resize_image = trans_Resize(tensor_image)  # 对image进行resize
writer.add_image("Resize", resize_image, 1)

# Compose(Resize+totensor)
trans_Resize_2 = transforms.Resize(720)
# Compose的作用是将多个transforms结合在一起diy一个流水线式工具[trans1,trans2...]让image依次经过不同的transforms，上一个transforms的输出就是下一个的输入
trans_compose = transforms.Compose([trans_Resize_2, trans_totensor])  # PIL Image-->PIL Image-->Tensor
resize_image_2 = trans_compose(image)
writer.add_image("Resize", resize_image_2, 2)

# RandomCrop
trans_randomcrop = transforms.RandomCrop(512)  # 随机得到一个短边修剪到512，长宽比不变的图像
trans_compose_2 = transforms.Compose([trans_randomcrop, trans_totensor])# 将RamdomCrop和Totensor结合起来，最终得到tensor数据类型
for i in range(10):
    randomcrop_image = trans_compose_2(image)
    writer.add_image("RandomCrop", randomcrop_image, i)

writer.close()
