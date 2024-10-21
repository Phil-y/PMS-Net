
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
import Config as config

'''
这段代码定义了一个名为random_rot_flip的函数，该函数接受一个图像（image）和一个标签（label）作为输入，并对图像和标签进行随机旋转和翻转操作。
'''
def random_rot_flip(image, label):
    # 使用np.random.randint(0, 4)随机生成一个0到3之间的整数k，表示旋转的次数
    k = np.random.randint(0, 4)
    # 使用np.rot90(image, k)和np.rot90(label, k)对图像和标签进行k次90度的逆时针旋转。
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    # print("image shape:", image.shape)
    # print("label shape:", label.shape)
    # 使用np.random.randint(0, 2)随机生成一个0或1的整数axis
    axis = np.random.randint(0, 2)
    # print("axis shape:", axis)
    # 使用np.flip(image, axis=axis)和np.flip(label, axis=axis)在axis轴上翻转图像和标签。
    # 使用.copy()方法创建这些翻转后的图像和标签的副本，以避免原始数据被修改
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label
'''random_rot_flip函数的功能是：随机对输入的图像和标签进行旋转和翻转操作，并返回处理后的图像和标签'''

'''
这段代码定义了一个名为random_rotate的函数，该函数接受一个图像（image）和一个标签（label）作为输入，并对图像和标签进行随机旋转操作
'''
def random_rotate(image, label):
    # 使用np.random.randint(-20, 20)随机生成一个-20到19之间的整数angle，表示旋转的角度
    angle = np.random.randint(-20, 20)
    # 使用ndimage.rotate(image, angle, order=0, reshape=False)和ndimage.rotate(label, angle, order=0, reshape=False)对图像和标签进行angle度的旋转。
    # order=0表示旋转时不插值
    # reshape=False表示不改变输出图像的形状，即输出图像的大小与输入图像相同
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    # 返回经过随机旋转操作后的图像和标签
    return image, label



'''这段代码定义了一个名为RandomGenerator的类，用于创建一个随机图像生成器。这个生成器可以随机对输入的图像进行旋转、翻转和缩放等操作，并将其调整到指定的输出尺寸。'''
class RandomGenerator(object):
    # __init__方法用于初始化类的实例。在这里，它接受一个参数output_size，并将其存储在实例变量self.output_size中
    def __init__(self, output_size):
        self.output_size = output_size
    # __call__方法使得对象可以像函数一样被调用。它接受一个参数sample，这里假设sample是一个包含图像和标签的字典
    def __call__(self, sample):
        # 从输入的sample中提取图像和标签数据
        image, label = sample['image'], sample['label']
        # print("image shape:",image.shape)
        # print("label shape:",label.shape)
        # 使用F.to_pil_image函数将图像和标签转换为PIL图像格式
        image, label = F.to_pil_image(image), F.to_pil_image(label)

        # print("Fimage :", image)
        # print("Flabel :", label)
        x, y = image.size
        # 如果随机数大于0.5，则对图像进行随机旋转和翻转操作。
        # 如果随机数小于0.5，则对图像进行随机旋转操作
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() < 0.5:
            image, label = random_rotate(image, label)

        # 如果图像的尺寸不等于指定的输出尺寸，则使用zoom函数将图像和标签调整到指定尺寸。
        # order=3用于图像插值，order=0用于标签插值
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # 使用F.to_tensor函数将图像转换为张量。
        # 使用to_long_tensor函数将标签转换为长整型张量
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        # print("image shape:", image.shape)
        # print("label shape:", label.shape)
        # 将处理后的图像和标签构建为一个字典，并返回该字典作为处理后的样本
        sample = {'image': image, 'label': label}
        # print(sample.keys(),sample.values())

        return sample
'''RandomGenerator类的功能是：根据给定的输出尺寸对输入的图像进行随机旋转、翻转和缩放等操作，并将其转换为张量形式'''


'''
这段代码定义了一个名为ValGenerator的类，用于创建一个验证数据生成器。
这个生成器的主要功能是将输入的图像和标签调整到指定的输出尺寸，并将它们转换为张量形式
'''
# 定义一个名为ValGenerator的类，它继承自Python中的object类
class ValGenerator(object):
    # __init__方法用于初始化类的实例。在这里，它接受一个参数output_size，并将其存储在实例变量self.output_size中
    def __init__(self, output_size):
        self.output_size = output_size
    # __call__方法使得对象可以像函数一样被调用。它接受一个参数sample，这里假设sample是一个包含图像和标签的字典
    def __call__(self, sample):
        # 从输入的sample中提取图像和标签数据
        image, label = sample['image'], sample['label']
        # 使用F.to_pil_image函数将图像和标签转换为PIL图像格式
        image, label = F.to_pil_image(image), F.to_pil_image(label)

        x, y = image.size
        # 如果图像的尺寸不等于指定的输出尺寸，则使用zoom函数将图像和标签调整到指定尺寸
        # order=3用于图像插值，order=0用于标签插值
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        #     使用F.to_tensor函数将图像转换为张量。
        # 使用to_long_tensor函数将标签转换为长整型张量
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        # 将处理后的图像和标签构建为一个字典，并返回该字典作为处理后的样本
        sample = {'image': image, 'label': label}
        return sample

'''to_long_tensor函数用于将输入的图片pic转换为长整型张量'''
def to_long_tensor(pic):
    # handle numpy array
    # 首先，使用np.array(pic, np.uint8)将图片pic转换为NumPy数组，并将其转换为无符号8位整型
    # 然后，使用torch.from_numpy()将NumPy数组转换为PyTorch张量。
    img = torch.from_numpy(np.array(pic, np.uint8))
    # 最后，使用.long()方法将张量转换为长整型。
    return img.long()


'''# correct_dims函数用于处理输入的图像列表images，确保每个图像都具有正确的维度'''
def correct_dims(*images):
    # 创建一个空列表corr_images来存储处理后的图像
    corr_images = []
    # print(images)
    # 然后，遍历输入的图像列表images
    for img in images:
        # 如果图像的维度为2（即灰度图像），使用np.expand_dims(img, axis=2)在第三个维度上增加一个维度，使其变为三维。
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        # 如果图像的维度不为2（即彩色图像或其他类型的图像），直接将其添加到corr_images列表中。
        else:
            corr_images.append(img)
    # 最后，如果corr_images列表中只包含一个图像，则返回该图像；否则，返回corr_images列表。
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

'''用于读取图像数据集并应用数据增强转换'''
# 定义一个名为ImageToImage2D的类，该类继承自Dataset
class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """
    # 这部分是构造函数的参数说明
    # dataset_path: 数据集路径。
    # joint_transform: 数据增强转换。
    # one_hot_mask: 是否返回独热编码形式的掩码。
    # image_size: 图像的尺寸，默认为224x224

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False, image_size: int =224) -> None:
        # 初始化类的属性
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask
        # 如果提供了joint_transform，则使用给定的增强转换；否则，使用ToTensor将图像和掩码转换为张量。
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        # 返回数据集的长度，即图像的数量
        return len(os.listdir(self.input_path))
    # 获取给定索引idx的图像和掩码
    def __getitem__(self, idx):
        # 获取指定索引的图像文件名
        image_filename = self.images_list[idx]
        # print(image_filename[: -4])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        # 读取并调整图像的尺寸
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        # print("image_filename",image_filename)
        # print("1",image.shape)
        image = cv2.resize(image,(self.image_size,self.image_size))
        # print(np.max(image), np.min(image))
        # print("2",image.shape)
        # read mask image

        # 根据config.task_name选择正确的掩码文件，并读取该掩码
        if config.task_name == "ISIC2017":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -4] + "_segmentation." + "jpg"),0)
        elif config.task_name == "BUSI_with_GT":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -4] + "_mask." + "png"), 0)
        elif config.task_name == "CHASE":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -4] + "_1stHO." + "png"), 0)
        elif config.task_name == "Chest Xray":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -4] + "_mask." + "png"), 0)
        elif config.task_name == "CVC-ClinicDB":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "tif"), 0)
        elif config.task_name == "Kvasir-Seg":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "jpg"), 0)
        elif config.task_name == "TG3K":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "jpg"), 0)
        elif config.task_name == "TN3K":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "jpg"), 0)
        else:
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"), 0)



        # print("mask",image_filename[: -3] + "png")
        # print(np.max(mask), np.min(mask))
        # 调整掩码尺寸并将其转换为二值掩码
        mask = cv2.resize(mask,(self.image_size,self.image_size))
        # print(mask.shape)
        # print(np.max(mask), np.min(mask))
        mask[mask<=0] = 0
        # (mask == 35).astype(int)
        mask[mask>0] = 1
        # print("11111",np.max(mask), np.min(mask))

        # 使用correct_dims函数确保图像和掩码具有正确的维度
        image, mask = correct_dims(image, mask)
        # image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # print("11",image.shape)
        # print("22",mask.shape)
        # 创建一个字典sample来存储图像和掩码
        sample = {'image': image, 'label': mask}
        # 如果提供了joint_transform，则应用数据增强转换
        if self.joint_transform:
            sample = self.joint_transform(sample)
        # sample = {'image': image, 'label': mask}
        # print("2222",np.max(mask), np.min(mask))
        # 如果one_hot_mask为True，则将掩码转换为独热编码形式
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print("mask",mask)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)
        # print(sample['image'].shape)
        # 返回处理后的数据和图像文件名
        return sample, image_filename






class ImageToImage2D_draw(Dataset):

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False, image_size: int =224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask




        img_names = os.listdir(dataset_path + '/img')
        img_names = sorted(img_names, key=lambda i: int(i.split(".")[0]))
        self.img_names = img_names






        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]
        #print(image_filename[: -3])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        # print("img",image_filename)
        # print("1",image.shape)
        image = cv2.resize(image,(self.image_size,self.image_size))
        # print(np.max(image), np.min(image))
        # print("2",image.shape)
        # read mask image

        if config.task_name == "ISIC2017":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "_segmentation" + "jpg"), 0)
        elif config.task_name == "BUSI_with_GT":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "_mask" + "png"), 0)
        elif config.task_name == "CHASE":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "_1stHO" + "png"), 0)
        elif config.task_name == "Chest Xray":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "_mask" + "png"), 0)
        elif config.task_name == "CVC-ClinicDB":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "tif"), 0)
        elif config.task_name == "Kvasir-Seg":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "jpg"), 0)
        elif config.task_name == "TG3K":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "jpg"), 0)
        elif config.task_name == "TN3K":
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "jpg"), 0)
        else:
            mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"), 0)


        # print("mask",image_filename[: -3] + "png")
        # print(np.max(mask), np.min(mask))
        mask = cv2.resize(mask,(self.image_size,self.image_size))
        # print(np.max(mask), np.min(mask))
        mask[mask<=0] = 0
        # (mask == 35).astype(int)
        mask[mask>0] = 1
        # print("11111",np.max(mask), np.min(mask))

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        # image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # print("11",image.shape)
        # print("22",mask.shape)
        sample = {'image': image, 'label': mask}

        if self.joint_transform:
            sample = self.joint_transform(sample)
        # sample = {'image': image, 'label': mask}
        # print("2222",np.max(mask), np.min(mask))

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)


        img_name = self.img_names[idx]
        image_path = os.path.join(self.dataset_path + '/img', img_name)
        label_path = os.path.join(self.dataset_path + '/labelcol', img_name)
        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
        assert os.path.exists(label_path), ('{} does not exist'.format(label_path))
        label_name = os.path.basename(label_path)
        sample['label_name'] = label_name
        return sample