import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pandas.util
import matplotlib.pyplot as plt
import numpy as np
#import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            # 卷积层R1：输入通道为1，输出通道为6，卷积核为5，步长为1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            # 池化层P1：过滤器大小为2*2
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            # 卷积层R2：输入通道为6，输出通道为16，卷积核为5，步长为1
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            # 池化层P2：过滤器大小为2*2
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            # 线性层L1：输入为非线性256，输出为线性120
            nn.Linear(in_features=4 * 4 * 16, out_features=120)
        )
        self.fc2 = nn.Sequential(
            # 线性层L2：输入为120，输出为84
            nn.Linear(in_features=120, out_features=84)
        )
        self.fc3 = nn.Sequential(
            # 线性层L3：输入为84，输出为10
            nn.Linear(in_features=84, out_features=10)
        )

    # 正向传播过程
    def forward(self, input):
        conv1_output = self.conv1(input)  # [28,28,1]-->[24,24,6]-->[12,12,6]
        conv2_output = self.conv2(conv1_output)  # [12,12,6]-->[8,8,16]-->[4,4,16]
        conv2_output = conv2_output.view(-1, 4*4*16)  # [n,4,4,16]-->[n,4*4*16],其中n代表个数
        fc1_output = self.fc1(conv2_output)  # [n,256]-->[n,120]
        fc2_output=self.fc2(fc1_output)  # [n,120]-->[n,84]
        fc3_output = self.fc3(fc2_output)  # [n,84]-->[n,10]
        return fc3_output
# 模型建立好了，下面进行训练
##############################################################

train_data = pd.DataFrame(pd.read_csv(r'D:\Code\ketang\dataset\mnist_dataset_csv\mnist_train.csv'))
model = LeNet()
loss_fc = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
optimizer = optim.SGD(params=model.parameters(), lr=0.001)  # 采用随机梯度下降SGD，参数lr为学习率
loss_list = []  # 记录每次的损失值
x = []  # 记录训练次数
for i in range(900):
    # 每次随机读取30条数据 sample(序列a，n)功能：从序列a中随机抽取n个元素，并将n个元素生以list形式返回
    batch_data = train_data.sample(n=30, replace=False)  
    batch_y = torch.from_numpy(batch_data.iloc[:, 0].values).long()  # 标签值
    batch_x = torch.from_numpy(batch_data.iloc[:, 1::].values).float().view(-1, 1, 28, 28)
    # 图片信息，一条数据784维将其转化为通道数为1， 大小为28*28的图片

    prediction = model.forward(batch_x)  # 向前传播
    loss = loss_fc(prediction, batch_y)  # 计算损失值

    optimizer.zero_grad()  # 将梯度置为零，在每次进行误差反传时应该先置零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重
    print('第%d次训练, loss=%.3f' % (i, loss))
    loss_list.append(loss)
    x.append(i)
torch.save(model.state_dict(), 'LeNet.pkl')  # 保存模型参数
plt.figure()
plt.plot(x, loss_list, 'r-')  # 可以将损失值进行绘制
plt.show()
#############在下一个文件里将训练好的模型用于测试数据进行测测试

model = LeNet()
test_data = pd.DataFrame(pd.read_csv(r'D:\Code\ketang\dataset\mnist_dataset_csv\mnist_test.csv'))
model.load_state_dict(torch.load('LeNet.pkl'))  # 加载模型参数
with torch.no_grad():  # 测试不需要反向传播
    batch_data = test_data.sample(n=50, replace=False)
    batch_x = torch.from_numpy(batch_data.iloc[:, 1::].values).float().view(-1, 1, 28, 28)
    batch_y = batch_data.iloc[:, 0].values
    prediction = np.argmax(model(batch_x).numpy(), axis=1)  # 在pytorch中.numpy()的意思是将tensor转化为numpy
    print(prediction)
    for i in range(100):
        batch_data = test_data.sample(n=50, replace=False)
        batch_x = torch.from_numpy(batch_data.iloc[:,1::].values).float().view(-1,1,28,28)
        batch_y = batch_data.iloc[:, 0].values
        prediction = np.argmax(model(batch_x).numpy(),axis=1)
        acccurcy = np.mean(prediction == batch_y)
        print("第%d组测试集，准确率为%.3f" % (i, acccurcy))
