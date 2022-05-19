import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

trans_ = torchvision.transforms.ToTensor()
data_train = torchvision.datasets.MNIST(root="./minist", train=True, transform=trans_, download=True)
data_test = torchvision.datasets.MNIST(root="./minist", train=False, transform=trans_, download=True)
loader = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
writer=SummaryWriter("Mnist")
step=0
for data in loader:
    images, targes = data
    writer.add_images("Train",images,step)
    step+=1

writer.close()