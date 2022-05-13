from torchvision.datasets import CIFAR10
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

test_data = CIFAR10(root="./torch_vision_dataset", train=False, transform=torchvision.transforms.ToTensor(),download=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# test_data中的第一张照片
image, target = test_data[0]
print(image)
print(image.shape)  # torch.Size([3, 32, 32])
print(target)

# Dataloader
writer = SummaryWriter("Dataloader")
for epoch in range(2):
    step = 0
    for data in test_data_loader:
        images, targets = data
        # print(images.shape)  # torch.Size([4, 3, 32, 32]) 4表示一个batch里面有四张照片
        # print(targets)  # tensor([6, 4, 7, 0])
        writer.add_images("Epoch：{}".format(epoch), images, step)
        # writer.add_images()表示加载多个照片
        step += 1

writer.close()
