import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 得到数据
dataset = torchvision.datasets.CIFAR10(root="./conv2d_dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
# 构建dataloader
dataloader = DataLoader(dataset, batch_size=64)


# 自定义卷积层
class my_Conv2d(nn.Module):
    def __init__(self):
        super(my_Conv2d, self).__init__()  # 初始化父类
        # 自定义一个卷积函数
        self.conv2d = Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=(1, 1), padding=0)

    def forward(self, x):
        x = self.conv2d(x)
        return x


# 得到一个卷积层对象
my_conv2d = my_Conv2d()
writer = SummaryWriter("Conv2d")
step = 0
for data in dataloader:
    images, targets = data
    output = my_conv2d(images)
    print(images.shape)  # torch.Size([64, 3, 32, 32])
    print(output.shape)  # torch.Size([64, 6, 30, 30])
    writer.add_images("Input", images, step)
    output = torch.reshape(output, [-1, 3, 30, 30])
    # 因为tensorboard不知道怎么去显示六个通道的图片，所以要进行reshape得到三通道图片
    # [-1,xx,xx,xx] 用-1就能让他自己计算决定多少个batch_size
    writer.add_images("Output", output, step)
    step += 1
writer.close()
