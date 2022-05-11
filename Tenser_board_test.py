from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
# add_scaler()
# #  y=x
# for i in range(-99,100):
#     writer.add_scalar("y=x^2", scalar_value=i*i, global_step=i)
# add_image()
img_path = "dataset/train/bees_image/90179376_abc234e5f4.jpg"
img = np.array(Image.open(img_path))
writer.add_image("Test",img,1,dataformats="HWC")
writer.close()
