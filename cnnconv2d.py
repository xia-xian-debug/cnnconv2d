import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import torch.optim as optim
import psutil  # 用于监控内存使用情况

# 定义CNN模型
class SimpleCnn(nn.Module):
    def __init__(self, inputChannels, numClasses):
        super(SimpleCnn, self).__init__()
        self.conv1 = nn.Conv2d(inputChannels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(128, numClasses)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 检查哪个设备可用
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'使用设备: {device}')

def loadCoraData(contentPath, citesPath):
    featuresList = []
    labelsList = []  # 创建两个空列表用于接收特征以及类别
    labelToInt = {}  # 在cora的数据集中标签为字符串，要转化为int形式
    labelValue = 0
    nodeToIndex = {}  # 用于将论文ID映射到索引
    currentIndex = 0

    # 从磁盘中加载数据到内存中
    with open(contentPath, 'r') as f:
        data = f.readlines()  # 读取文件每一行
    for line in data:
        parts = line.strip().split()  # 将每一行的元素按空格分割
        nodeId = parts[0]  # 论文ID
        if nodeId not in nodeToIndex:
            nodeToIndex[nodeId] = currentIndex
            currentIndex += 1
        featuresList.append(list(map(float, parts[1:-1])))
        label = parts[-1]
        if label not in labelToInt:  # 该标签在字典中无对应值
            labelToInt[label] = labelValue
            labelValue += 1
        labelsList.append(labelToInt[label])
    # 转化为numpy数组
    featuresList = np.array(featuresList)
    labelsList = np.array(labelsList)

    edges = []
    with open(citesPath, 'r') as f:
        data = f.readlines()
    for line in data:
        parts = line.strip().split()
        node1 = parts[0]  # 论文ID
        node2 = parts[1]  # 论文ID
        if node1 in nodeToIndex and node2 in nodeToIndex:  # 确保节点ID在映射中
            edges.append((nodeToIndex[node1], nodeToIndex[node2]))
        else:
            print(f"警告: 忽略无效的边 ({node1}, {node2})，节点ID未找到")

    # 构建邻接矩阵
    numNodes = len(featuresList)
    adjMatrix = np.zeros((numNodes, numNodes))
    for edge in edges:
        adjMatrix[edge[0], edge[1]] = 1
        adjMatrix[edge[1], edge[0]] = 1  # 无向图
    return featuresList, labelsList, adjMatrix

def convertToCnnFormat(featuresList, labelsList):
    numNodes, numFeatures = featuresList.shape  # 提取出输入样本数以及特征类别总数
    featuresList = featuresList.reshape(numNodes, 1, numFeatures, 1)
    featuresList = torch.tensor(featuresList, dtype=torch.float32)  # 只有转为float32才可以被加载器识别
    labelsList = torch.tensor(labelsList, dtype=torch.long)
    return featuresList, labelsList, numNodes, numFeatures


# 加载数据
featuresList, labelsList, adjMatrix = loadCoraData('/home/xiaxia/桌面/cora/cora.content', '/home/xiaxia/桌面/cora/cora.cites')
featuresList, labelsList, numNodes, numFeatures = convertToCnnFormat(featuresList, labelsList)

# 将模型移动到设备上
model = SimpleCnn(1, 7).to(device)

# 划分数据集
idxArray = torch.arange(numNodes)  # 创建一个0开始的索引数组，便于后面打包特征和标签
trainIdx, testIdx = train_test_split(idxArray, train_size=0.8, test_size=0.2, random_state=17)
testIdx, verifyIdx = train_test_split(testIdx, train_size=0.5, test_size=0.5, random_state=17)  # 划分训练集，测试集，验证集为8：1：1

# 创建数据加载器
trainFeatures = torch.index_select(featuresList, 0, trainIdx).to(device)
trainLabels = torch.index_select(labelsList, 0, trainIdx).to(device)
testFeatures = torch.index_select(featuresList, 0, testIdx).to(device)
testLabels = torch.index_select(labelsList, 0, testIdx).to(device)
verifyFeatures = torch.index_select(featuresList, 0, verifyIdx).to(device)
verifyLabels = torch.index_select(labelsList, 0, verifyIdx).to(device)

# 包装数据特征和标签
trainDataset = TensorDataset(trainFeatures, trainLabels)
testDataset = TensorDataset(testFeatures, testLabels)
verifyDataset = TensorDataset(verifyFeatures, verifyLabels)

trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=32, shuffle=False)
verifyLoader = DataLoader(verifyDataset, batch_size=32, shuffle=False)

# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

epochs = 50

# 记录初始内存和时间
startTime = time.time()
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清空缓存
    initialMemory = torch.cuda.memory_allocated() / 1024  # 初始分配的显存（KB）
else:
    process = psutil.Process(os.getpid())
    initialMemory = process.memory_info().rss / 1024  # 初始内存（KB）

for epoch in range(epochs):
    model.train()
    for idx, (data, target) in enumerate(trainLoader):
        data, target = data.to(device), target.to(device)  # 将数据移动到设备上
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    model.eval()
    verifyLoss = 0
    correct = 0
    with torch.no_grad():
        for data, target in verifyLoader:
            data, target = data.to(device), target.to(device)  # 将数据移动到设备上
            output = model(data)
            verifyLoss += criterion(output, target).item() * data.size(0)  # 计算损失
            predict = output.argmax(dim=1, keepdim=True)  # 获取预测结果
            correct += predict.eq(target.view_as(predict)).sum().item()
    verifyLoss /= len(verifyLoader.dataset)
    accuracy = 100. * correct / len(verifyLoader.dataset)
    print(f'训练周期{epoch + 1}，损失值{verifyLoss:.4f}，准确值{accuracy:.2f}%')

# 记录训练结束时间
endTime = time.time()
trainingTime = endTime - startTime
print(f'训练时间: {trainingTime:.2f} 秒')

# 记录训练结束后的内存占用
if torch.cuda.is_available():
    finalMemory = torch.cuda.memory_allocated() / 1024  # 最终分配的显存（KB）
    memoryUsage = finalMemory - initialMemory
    print(f'GPU 内存占用: {memoryUsage:.2f} KB')
else:
    process = psutil.Process(os.getpid())
    finalMemory = process.memory_info().rss / 1024  # 最终内存（KB）
    memoryUsage = finalMemory - initialMemory
    print(f'CPU 内存占用: {memoryUsage:.2f} KB')

model.eval()
testLoss = 0
correct = 0
with torch.no_grad():
    for data, target in testLoader:
        data, target = data.to(device), target.to(device)  # 将数据移动到设备上
        output = model(data)
        testLoss += criterion(output, target).item() * data.size(0)
        predict = output.argmax(dim=1, keepdim=True)
        correct += predict.eq(target.view_as(predict)).sum().item()
testLoss /= len(testLoader.dataset)
accuracy = 100. * correct / len(testLoader.dataset)
print(f'损失值{testLoss:.4f}，准确值{accuracy:.2f}%')