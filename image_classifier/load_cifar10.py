"""
加载cifar10的数据以便i其他代码调用
"""
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from  PIL import Image
import glob
import os


label_name = ["airplane",
                        "automobile",
                        "bird",
                        "cat",
                        "deer",
                        "dog",
                        "frog",
                        "horse",
                        "ship",
                        "truck",
                        ]

# 把标签转成一个字典，便于后面使用
label_dict = dict()
for idx, name in enumerate(label_name):
    label_dict[name] = idx


def default_loader(path):
    """
    :param  path: str, 
    读取图片
    """
    return Image.open(path).convert('RGB')


# 训练集的增强函数
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(90),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
    # transforms.RandomGrayscale(0.2),
    # transforms.RandomCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# 测试集的增强函数
test_transform = transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


class Cifar10Dataset(Dataset):
    def __init__(self, im_list, transform=None, loader=default_loader):
        super().__init__()

        imgs = list()
        for im_item in im_list:
            #"/path/to/CIFAR10/TRAIN/airplane/aeroplane_s_000021.png"
            im_label_name = im_item.split("\\")[-2]
            # imgs是一个list，每个元素也是一个包含两个元素的list，第一个是图像的路径，第二个的图像标签
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index) :
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)

        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


im_train_list = glob.glob(os.path.abspath(os.path.dirname(__file__)) +  '\TRAIN\*\*.png')
im_test_list = glob.glob(os.path.abspath(os.path.dirname(__file__)) +  '\TEST\*\*.png')

train_dataset = Cifar10Dataset(im_train_list, transform=train_transform)
test_dataset = Cifar10Dataset(im_test_list, transform=test_transform)

# 当DataLoader的num_works参数大于0时，需要把dataloader挡在main函数中运行，否则会报错
# 参见： http://www.cxyzjd.com/article/Elvirangel/101076930
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=4)

# print("num_of_train", len(train_dataset))
# print("num_of_test", len(test_dataset))
