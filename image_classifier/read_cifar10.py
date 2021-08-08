"""
读取cifar10的数据并分类存放，供其他代码使用
"""
import glob
import pickle
import numpy as np
import os
import cv2


# cifar10 中的全部分类
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

# 获取训练数据的列表
train_list = glob.glob(".\cifar-10-batches\data_batch_*")
# 获取测试数据的列表
test_list = glob.glob(".\cifar-10-batches\\test_batch*")
# 训练集的存放路径
train_data_path = os.path.abspath(os.path.dirname(__file__)) +  '\TRAIN'
# 测试集的存放路径
test_data_path = os.path.abspath(os.path.dirname(__file__)) +  '\TEST'


def unpickle(file):
    """
    获取每一个batch的数据并返回一个字典
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_image(label_name, unpickle, train_list, train_data_path):
    for item in train_list:
        item_dict = unpickle(item)

        for im_idx, im_data in enumerate(item_dict[b'data']):
            im_label = item_dict[b'labels'][im_idx]
            im_name = item_dict[b'filenames'][im_idx]
        
            im_label_name = label_name[im_label]
            im_data = np.reshape(im_data, [3, 32, 32])
            im_data = np.transpose(im_data, (1, 2, 0))

        # 展示图片
        # cv2.imshow("im_data", cv2.resize(im_data, (100, 100)))
        # cv2.waitKey(0)

            if not os.path.exists("{}/{}".format(train_data_path,  im_label_name)):
                os.mkdir("{}/{}".format(train_data_path, im_label_name))
        
        # 把图片按照分类存放
            cv2.imwrite('{}/{}/{}'.format(train_data_path, im_label_name, im_name.decode('utf-8')), im_data)

def main():
    # TODO: 当数据量特别大的时候这里可以做多线程加速过程
    get_image(label_name, unpickle, train_list, train_data_path)
    get_image(label_name, unpickle, test_list, test_data_path)


if __name__ == '__main__':
    main()
