"""
训练脚本
"""
import torch
torch.cuda.empty_cache()
from torch import optim
import torch.nn as nn
import torchvision
from vggnet import vggNet
from load_cifar10 import train_loader, test_loader
import os
import tensorboardX


def tranning():
    """
    训练函数
    """
    # 选定GPU还是CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = vggNet().to(device)

    epochs = 200
    lr = 0.01
    batch_size = 128

    # 多分类问题，采用交叉熵loss
    loss_func = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 动态学习率
    scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # 设定tensorboard日志的路径
    if not os.path.exists('log'):
        os.mkdir('log')
    writer = tensorboardX.SummaryWriter('log')

    for epoch in range(epochs):
        epoch += 1
        print('epoch : ', epoch)

        ## 训练
        # 计算每个epoch的损失和正确率
        train_sum_loss = 0
        train_sum_correct = 0
        for idx, data in enumerate(train_loader):
            net.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
             # 更新参数
            optimizer.step()

            # print('train step: ', idx)
            # print("loss is: ", loss.item())

            # 计算每个batch_size的正确率
            _, pred = torch.max(outputs.data, dim=1)
            correct  = pred.eq(labels.data).cpu().sum()
            train_sum_loss += loss.item()
            train_sum_correct += correct.item()
        train_loss = train_sum_loss * 1.0 / len(train_loader)
        train_correct = train_sum_correct * 100.0 / len(train_loader) / batch_size
        print('epoch: ', epoch, '; train loss : ', train_loss, '; train correct: ', train_correct)

        # 添加日志到tensorboard中
        writer.add_scalar('train loss', train_loss.item(), global_step=epoch)
        writer.add_scalar('train correct', train_correct, global_step=epoch)
        
        # 按照每一个epoch保存模型
        # if not os.path.exists('models'):
        #     os.mkdir('models')
        # torch.save(net.state_dict(), 'models/{}.path'.format(epoch))
        
        # 一个epoch结束后更新学习率
        scheduler.step()
        
        ## 测试 ##
        # 计算每个epoch的损失和正确率
        test_sum_loss = 0
        test_sum_correct = 0
        for idx, data in enumerate(test_loader):
            net.eval()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
             # 更新参数
            optimizer.step()

            # print('test step: ', idx)
            # print("loss is: ", loss.item())

            # 计算正确率
            _, pred = torch.max(outputs.data, dim=1)
            correct  = pred.eq(labels.data).cpu().sum()

            test_sum_loss += loss.item()
            test_sum_correct += correct.item()
        test_loss = test_sum_loss * 1.0 /len(test_loader)
        test_correct = test_sum_correct * 100.0 /len(test_loader) / batch_size
        # print('epoch: ', epoch, '; test loss : ', test_loss, '; test correct: ', test_correct)
        
        # 添加日志到tensorboard中
        writer.add_scalar('test loss', test_loss.item(), global_step=epoch)
        writer.add_scalar('test correct', test_correct, global_step=epoch)
        

def main():
    """
    把训练过程放在main函数中，
    这样在dataloader的num_works大于0的时候才不会报错
    """
    tranning()


if __name__ == '__main__':
    main()
