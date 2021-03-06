# 两个重要的函数

* dir()
* help()

![image-20220508023940432](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205080240499.png)

将pytorch想象成一个大的工具箱，工具箱中有多个分区装着不同的工具

通过函数dir(pytorch)函数可以知道工具箱中有哪些分区eg：1，2，3...

通过函数dir(pytorch.区域x)函数就可以知道区域x有哪些函数(工具)eg：a,b,c...

而help()函数可以具体告诉我们这个函数怎么使用(参数，返回值啥的)eg:help(pytorch.3,a)就告诉我们3号区域的a函数怎么使用

![image-20220508025052941](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205080250004.png)

![image-20220508025101376](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205080251426.png)



![image-20220508030811444](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205080308608.png)

# 加载数据

## Dataset

提供一种方法来获取数据集中的数据以及数据集每个数据的label

```python
from torch.utils.data import Dataset
#从torch.utils.data导入Dataset类
```

```python
class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py
```

这告诉我们继承这个Dataset类需要重写Dataset类中的__getitem\__函数

```python
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
```

init就是初始化各个路径便于后面获取一张图片，所以我们要先获取所有照片的地址，因此我们生成一个照片的地址的列表，这些地址只是相对地址

![image-20220510112651537](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205101126570.png)

![image-20220510112339966](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205101123037.png)

getitem就是我们根据索引在地址列表中找到照片的名字，然后用join合成得到照片的地址，之后我们就通过Image.open(image_path)得到这个照片，最后将这个照片以及标签返回

返回的照片是一个Image对象，包含很多信息

![image-20220510113240186](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205101132235.png)

len（）函数就是根据照片地址列表成员的数量返回这个数据集有多大

```python
root_dir="dataset/train"
label_dir="ants_image"
ants_dataset=Mydata(root_dir,label_dir)
root_dir="dataset/train"
label_dir="bees_image"
bees_dataset=Mydata(root_dir,label_dir)
train_dataset=ants_dataset+bees_dataset
```

获取两个数据集，同时我们也可以将两个数据集合并成一个traindataset

![image-20220510113115962](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205101131002.png)

![image-20220510113139069](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205101131095.png)

然后，在我们构建数据集的时候，一般会将数据和标签分开存储

![image-20220510113402744](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205101134841.png)

可以参考以下代码，将标签存在另一个标签文件夹中，存在txt当中

```python
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
```

重要的一些方法·函数

os

```python
os.path.join(path_a,path_b)#合成路径名称
os.listdir(path)#path目录中的全部路径组成的列表
```





## Dataloader

为后面的网络提供不同的数据类型,就是按照什么样的方式取出数据

```python
from torch.utils.data import DataLoader
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
```

<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205131837565.png" alt="image-20220513183758484" style="zoom:50%;" />

几个重要的参数解释：

* data数据集
* batch_size=表示每次取出多少个图片
* shuffle=True表示从数据集把所有数据取出来之后，下次取数据就会将所有数据打乱再去取，相当于洗牌
* num_workers表示进程数量，默认是0，表示只用主进程
* drop_last=True表示如果图片数量不能整除你的batch_size就不会将多余的图片取出来：如下图

<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205131831390.png" alt="image-20220513183131327" style="zoom: 50%;" />

<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205131832387.png" alt="image-20220513183226309" style="zoom: 50%;" />

普通的tensor数据类型的image：

```python
# test_data中的第一张照片
image, target = test_data[0]
print(image)
print(image.shape)  # torch.Size([3, 32, 32])
print(target) # 3
```

而dataloader得到的每个返回对象是一个batch_size数量的图片的集合

```python
print(images.shape)  # torch.Size([4, 3, 32, 32]) 4表示一个batch里面有四张照片
print(targets)  # tensor([6, 4, 7, 0])
```

完整的一个代码：

```python
from torchvision.datasets import CIFAR10
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

test_data = CIFAR10(root="./torch_vision_dataset", train=False, transform=torchvision.transforms.ToTensor())
#得到CIFAR10数据集，同时对数据进行tensor化
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
#得到一个DataLoader,batch_size设置为64，表示一次处理64张照片

# test_data中的第一张照片
#image, target = test_data[0]
#print(image)
#print(image.shape)  # torch.Size([3, 32, 32])
#print(target)

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
```











# TenserBoard

```python
from torch.utils.tensorboard import SummaryWriter
```

从tensorboard中导入SummaryWriter

```python
class SummaryWriter(object):
    """Writes entries directly to event files in the log_dir to be
    consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    Examples::

            from torch.utils.tensorboard import SummaryWriter

            # create a summary writer with automatically generated folder name.
            writer = SummaryWriter()
            # folder location: runs/May04_22-14-54_s-MacBook-Pro.local/

            # create a summary writer using the specified folder name.
            writer = SummaryWriter("my_experiment")
            # folder location: my_experiment

            # create a summary writer with comment appended.
            writer = SummaryWriter(comment="LR_0.1_BATCH_16")
            # folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/
    """
    
```

初始化一个SummaryWriter对象要用一个文件名来初始化，然后就会生成一个文件夹用来存之后的信息：作图啥的

just like logs



常用的是

```python
writer=SummaryWriter("文件名")#初始化一个w对象
writer.add_image()#添加图片
writer.add_scaler()#画图的
writer.close()
```

* add_scaler()画图的，之后可以直观的看到损失函数和训练步数之间的关系

  类似：

  <img src="C:/Users/LENOVO/AppData/Roaming/Typora/typora-user-images/image-20220511105755568.png" alt="image-20220511105755568" style="zoom:50%;" />

  ```python
  def add_scalar(
      self,
      tag,#图的标题 title
      scalar_value,# y轴的数值
      global_step=None,# 步长 x轴的数值坐标
      walltime=None,
      new_style=False,
      double_precision=False,
  ):
      """Add scalar data to summary.
  
      Args:
          tag (string): Data identifier
          scalar_value (float or string/blobname): Value to save
          global_step (int): Global step value to record
          walltime (float): Optional override default walltime (time.time())
            with seconds after epoch of event
          new_style (boolean): Whether to use new style (tensor field) or old
            style (simple_value field). New style could lead to faster data loading.
      Examples::
  
          from torch.utils.tensorboard import SummaryWriter
          writer = SummaryWriter()
          x = range(100)
          for i in x:
              writer.add_scalar('y=2x', i * 2, i)
          writer.close()
  
      Expected result:
  
      .. image:: _static/img/tensorboard/add_scalar.png
         :scale: 50 %
  
      """
  ```

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
#  y=x
for i in range(-99,100):
    writer.add_scalar("y=x^2", scalar_value=i*i, global_step=i)
writer.close()
```

运行完之后进入terminal输入命令：

```python
tensorboard --logdir=logs
logdir=文件夹的路径
```

![image-20220511110132265](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205111101315.png)

然后就可以在窗口进去看见图像了

如果说不修改图像tag，然后再次运行就可能会出现奇怪的图像，这是计算机帮忙拟合了，解决办法是，打开logs文件夹，删除里面的文件再次运行

![image-20220511110257926](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205111102955.png)

* add_image()

```python
def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
    """Add image data to summary.

    Note that this requires the ``pillow`` package.

    Args:
        tag (string): Data identifier
        img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
        global_step (int): Global step value to record
        walltime (float): Optional override default walltime (time.time())
          seconds after epoch of event
    Shape:
        img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
        convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
        Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
        corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.

    Examples::

        from torch.utils.tensorboard import SummaryWriter
        import numpy as np
        img = np.zeros((3, 100, 100))
        img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
        img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

        img_HWC = np.zeros((100, 100, 3))
        img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
        img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000

        writer = SummaryWriter()
        writer.add_image('my_image', img, 0)

        # If you have non-default dimension setting, set the dataformats argument.
        writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
        writer.close()

    Expected result:

    .. image:: _static/img/tensorboard/add_image.png
       :scale: 50 %

    """
```

```python
tag  表示图像的标题
img_tensor  表示图像信息，必须是tensor/numpy.ndarray/string类型
global_step  表示步长
```

读取图片

```python
imag_path="dataset/train/ants_image/0013035.jpg"
from PIL import Image
img=Image.open(imag_path)
```

然后我们会发现img的数据类型不是img_tensor所规定的数据类型

![image-20220511181239915](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205111812959.png)

一般情况下，我们会选择使用OpenCV读取图像，会直接获得Numpy型的图片数据

在这里我们直接强转

```python
import numpy as np
img_array=np.array(img)
print(type(img_array))
----->   <class 'numpy.ndarray'>
```

```python
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
img_path = "dataset/train/ants_image/20935278_9190345f6b.jpg"
img = np.array(Image.open(img_path))
writer.add_image("Image",img,1)
writer.close()

```

这样写会直接报错

![image-20220511182120797](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205111821854.png)

原因是对img_tensor的shape是有要求的

```python
Shape:
    img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
    convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
    Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
    corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.
```

默认的shape是（3，H，W）分别是通道数，高度，宽度，如果你的数据和默认不符合，就需要自己声明自己是什么样的shape,也就是参数dataformats=""

另外有三种shape：`CHW` `HWC` `HW`

```python
print(img_array.shape)
(512, 768, 3)
```

于是我们修改参数

```python
writer.add_image("Image",img,1,dataformats="HWC")
```

>从PIL到numpy,需要在add image()中,指定shape中每一个数字/维表示的含义

然后打开tensorboard就可以看到

![image-20220511182719563](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205111827648.png)

![image-20220511182837652](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205111828779.png)

使用tensorboard可以直观的看到训练的时候使用了哪些数据，训练了如何（损失值，训练次数...）

# Transforms

> 使用Transforms实现对图形的变换

* ## 如何使用transforms

整体的结构

![image-20220511184712131](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205111847214.png)

Transforms是一个大的工具箱，里面有很多工具，例如Totensor,resize...这些都是一个个不同的类

然后我们需要的是将这些用到的工具拿出来实例化一个工具对象，成为自己的工具

```python
from torchvision import transforms
from PIL import Image

img_path = "dataset/train/bees_image/39747887_42df2855ee.jpg"
img = Image.open(img_path)  # 得到一个PIL的Image
img_transforms = transforms.ToTensor()  # 实例化一个Totensor对象
img_tensor = img_transforms(img)  # 得到图像经过totensor的数据
print(img_tensor)

```

输出结果：

![image-20220511185308859](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205111853916.png)

* 为什么要使用tensor数据类型

  * 包含了神经网络之后需要的数据

    ![image-20220511185515279](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205111855347.png)

之前使用Tensorboard的时候，img_tensor就是需要tensor数据类型，我们重新实现一次

```python
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
```

**总结：Totensor使用步骤**

```python
#导入库
from torchvision import transforms
#实例化一个Totensor工具对象
tool=transforms.Totensor()
#将图片地址作为tool的参数，得到tensor数据类型的image
tensor_img=tool(img_path)
```

几个常用工具的功能：

```python
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

image_path = "Images/wallhaven.jpg"

writer = SummaryWriter("logs")
image = Image.open(image_path)  # PIl数据类型
print(image)
```

## ToTensor

```python
# Totensor
trans_totensor = transforms.ToTensor()  # 实例化Totensor
tensor_image = trans_totensor(image)  # 将PIL数据类型的照片装换成tensor类型
writer.add_image("Totensor", tensor_image)
```

## Resize

> 修改图片的尺寸
>
> * Resize((x,y)) 将长宽裁剪成（x,y）大小
> * Resize(x)将较短边裁剪成x，较长边进行“长宽比不变”的缩放

```python
# Resize
trans_Resize = transforms.Resize(512)  # 实例化一个Resize，参数是照片的大小，如果resize(512)表示将最短边调整至512，长宽比不变
# resize_image = trans_Resize(image)
resize_image = trans_Resize(tensor_image)  # 对image进行resize
writer.add_image("Resize", resize_image, 1)
```

（x,y）型裁剪：

<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205131446058.png" alt="image-20220513144558856" style="zoom:50%;" />

长宽比不变裁剪:

<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205131446290.png" alt="image-20220513144649175" style="zoom: 50%;" />

## Nomalize

> ## 归一化定义与作用
>
>   ==**归一化**就是要把需要处理的数据经过处理后（通过某种算法）限制在你需要的一定范围内。==首先归一化是为了后面数据处理的方便，其次是保证程序运行时收敛加快。归一化的具体作用是归纳统一样本的统计分布性。归一化在0-1之间是统计的概率分布，归一化在某个区间上是统计的坐标分布。归一化有同一、统一和合一的意思。
>
>   **归一化的目的**简而言之，是使得没有可比性的数据变得具有可比性，同时又保持相比较的两个数据之间的相对关系，如大小关系；或是为了作图，原来很难在一张图上作出来，归一化后就可以很方便的给出图上的相对位置等。
>
>   在使用机器学习算法的数据预处理阶段，归一化也是非常重要的一个步骤。例如在应用SVM之前，缩放是非常重要的。Sarle的神经网络FAQ的第二部分（1997）阐述了缩放的重要性，大多数注意事项也适用于SVM。缩放的最主要**优点是能够避免大数值区间的属性过分支配了小数值区间的属性**。另一个优点能**避免计算过程中数值复杂度**。因为关键值通常依赖特征向量的内积（inner products），例如，线性核和多项式核，属性的大数值可能会导致数值问题。我们推荐将每个属性线性缩放到区间[-1,+1]或者[0, 1]。
>
>   当然，我们**必须使用同样的方法缩放训练数据和测试数据。**例如，假设我们把训练数据的第一个属性从[-10,+10]缩放到[-1, +1]，那么如果测试数据的第一个属性属于区间[-11, +8]，我们必须将测试数据转变成[-1.1, +0.8]。

```python
class Normalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """
```

这个类的构造函数是：

```python
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        _log_api_usage_once(self)
        self.mean = mean
        self.std = std
        self.inplace = inplace
```

主要有两个参数：

mean=平均数

std=标准差

```python
# Nomalize

trans_nomalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 实例化Nomalize，初始化的参数就是平均值和标准差
nomal_image = trans_nomalize(tensor_image)
writer.add_image("Nomalize", nomal_image, 1)
print(tensor_image[0][0][0])  # 比较标准化之后的数据区别
print(nomal_image[0][0][0])
```

可以看到进行归一化之后的图片：

<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205131450810.png" alt="image-20220513145046702" style="zoom:50%;" />

## Compose

```python
class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """
```

Compose的参数是transforms，这个工具的作用是，将多个tansforms可以组合在一起，让"输入"依次经过Compose组合的transforms工具的操作，得到一个最终的输出

eg:

```python
# Compose(Resize+totensor)
trans_Resize_2 = transforms.Resize(720)
# Compose的作用是将多个transforms结合在一起diy一个流水线式工具[trans1,trans2...]让image依次经过不同的transforms，上一个transforms的输出就是下一个的输入
trans_compose = transforms.Compose([trans_Resize_2, trans_totensor])  # PIL Image-->PIL Image-->Tensor
resize_image_2 = trans_compose(image)
writer.add_image("Resize", resize_image_2, 2)
```

在这里我们将Resize和Totensor进行组合

首先我们将PIL Image输入，经过Resize得到缩放后的PIL Image，然后经过Totensor得到向量化之后的图片

这个就是一个==流水线式工作==





## RandomCrop

> 随机裁剪

```python
# RandomCrop
trans_randomcrop = transforms.RandomCrop(512)  # 随机得到一个短边修剪到512，长宽比不变的图像
trans_compose_2 = transforms.Compose([trans_randomcrop, trans_totensor])# 将RamdomCrop和Totensor结合起来，最终得到tensor数据类型
for i in range(10):
    randomcrop_image = trans_compose_2(image)
    writer.add_image("RandomCrop", randomcrop_image, i)
```

得到的结果：
<center>
<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205131457682.png" alt="image-20220513145718538" style="zoom:33%;" />
<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205131457852.png" alt="image-20220513145726716" style="zoom:33%;" />
<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205131457926.png" alt="image-20220513145732772" style="zoom:33%;" />
<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205131457339.png" alt="image-20220513145746191" style="zoom:33%;" />
</center>



# TorchVision中数据集的简单使用

导入数据集

```python
from torchvision.datasets import CIFAR10
from torchvision import transforms
dataset_transforms = transforms.Compose([
    transforms.ToTensor()
])
train_set = CIFAR10(root="./torch_vision_dataset", transform=dataset_transforms, train=True, download=True)
test_set = CIFAR10(root="./torch_vision_dataset", transform=dataset_transforms, train=False, download=True)
```

root表示存储地址

transform表示要对数据进行什么样的操作

train=True表示这是个训练集

download=True表示要进行下载（如果已经下载了也不胡报错）

```python
print(test_set[0])  
# (<PIL.Image.Image image mode=RGB size=32x32 at 0x1D6FD68EBC8>, 3)
#前一部分表示图片,3表示的是图片类别（target）
img, target = test_set[0]
print(img)
print(target)
print(test_set.classes)
```

转换成tensor之后就可以用tensorboard查看照片了

```python
writer = SummaryWriter("P10")
for i in range(100):
    image, target = train_set[i]
    writer.add_image("CIFAR10_train_Set", image, i)
writer.close()
```

<center class="half">
    <img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205131730154.png" alt="image-20220513173043110" style="zoom:40%;" /><img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205131729402.png" alt="image-20220513172959359" style="zoom:40%;" />
</center>


# 神经网络构建nn.Module

使用nn.Module来构建自己的自定义网络

```python
from torch import nn

class net(nn.Module):#首先要继承nn.Module这个类
	def __init__(self):
		super().__init__()#初始化必须先要使用父类的初始化
		
	def forward(self,x):
		#code
```

## 卷积操作(nn.functional)

nn里面有一些函数是包含在nn.functional中的

![image-20220517031905674](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205170319724.png)

==Convolution functions==卷积函数

![image-20220517031954229](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205170319271.png)

分别是一维，二维，三维的卷积函数

conv2d

> Parameters//一些参数

![image-20220517032314683](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205170323728.png)

input就是输入数据,对他的shape是有要求的分别是(batch_size,通道数,高度,宽度)

weight可以说是权重,也可以说是卷积核,同样的对他的shape是有要求的

bias就是偏置项

stride就是步长(可以观察卷积函数计算过程就知道是怎么一回事了)

padding就是填充,striding=1,就是将原始数据的周围填充一层0

![image-20220517032638999](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205170326061.png)

填充之后

![image-20220517032658815](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205170326863.png)

代码尝试:

```python
import torch
from torch import nn
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernal = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernal = torch.reshape(kernal, (1, 1, 3, 3))

output = F.conv2d(input, kernal, stride=1, padding=0)
print(output)
```

运行结果:

![image-20220517033353587](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205170333638.png)

```python
output2 = F.conv2d(input, kernal, stride=2, padding=0)
print(output2)
```

![image-20220517033439724](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205170334780.png)

```python
output3 = F.conv2d(input, kernal, stride=1, padding=1)
print(output3)
```

<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205170335331.png" alt="image-20220517033529284" style="zoom:67%;" />

![image-20220517034051571](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205170340655.png)

 

## 神经网络卷积层（nn）

![image-20220517174239057](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205171743451.png)

几个重要的参数

![image-20220517174501113](https://raw.githubusercontent.com/danjuan-77/picgo/main/202205171745192.png)

in_channels 表示输入的数据的通道数量

out_channels 表示输出的数据的通道数量

kernel_size表示你的卷积核的尺寸

stride表示每次的步长

padding表示周围填充

padding_mode表示按照什么样的方式填充



代码实战：

```python
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
```

将输出的图片加入到tensorboard中，要注意图片的通道数量

```python
output = torch.reshape(output, [-1, 3, 30, 30])
```

因为tensorboard不知道显示六通道数据

所以我们将他变成3通道的输出，这样就会改变batch_size的值，也就是说，通道数变少了，那就会把多出来的那几个通道，用多个batch_size显示出来

<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205171824432.png" alt="image-20220517182458352" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/danjuan-77/picgo/main/202205171825254.png" alt="image-20220517182514154" style="zoom:50%;" />















