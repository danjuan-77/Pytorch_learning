from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset_transforms = transforms.Compose([
    transforms.ToTensor()
])

train_set = CIFAR10(root="./torch_vision_dataset", transform=dataset_transforms, train=True, download=True)
test_set = CIFAR10(root="./torch_vision_dataset", transform=dataset_transforms, train=False, download=True)

# print(test_set[0])  # (<PIL.Image.Image image mode=RGB size=32x32 at 0x1D6FD68EBC8>, 3)前一部分表示图片,3表示的是图片类别（target）
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes)
writer = SummaryWriter("P10")
for i in range(100):
    image, target = train_set[i]
    writer.add_image("CIFAR10_train_Set", image, i)
writer.close()
