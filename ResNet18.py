import torch
import torch.nn as nn
import time
import onnx
from torch.nn import functional as F
import torchvision
import torch.utils.data as Data
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
from torch.quantization.quantize_fx import prepare_fx,convert_fx
import numpy
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = False # 如果你已经下载好了mnist数据就写上 False


# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:3000]

class ResnetBasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(ResnetBasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,stride,1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self,x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x+output)

class ResnetdownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(ResnetdownBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,stride[1],1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self,x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x+out)

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = nn.Sequential(ResnetBasicBlock(64,64,1),
                                    ResnetBasicBlock(64,64,1))

        self.layer2 = nn.Sequential(
            ResnetdownBlock(64,128,[2,1]),
            ResnetBasicBlock(128,128,1)
        )
        self.layer3 = nn.Sequential(
            ResnetdownBlock(128, 256, [2, 1]),
            ResnetBasicBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            ResnetdownBlock(256, 512, [2, 1]),
            ResnetBasicBlock(512, 512, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512,10)
    def forward(self,x):
        output = self.conv1(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avgpool(output)
        output = output.reshape(x.shape[0],-1)
        output = self.fc(output)
        return output

resnet =ResNet18()
optimizer = torch.optim.Adam(resnet.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader
        output = resnet(b_x)
        output = torch.squeeze(output)
        loss = loss_func(output, b_y.long())   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()
# resnet.load_state_dict(torch.load('resnet18.pth'))
resnet.eval()
start = time.time()
test_output = resnet(test_x)
end = time.time()
s = 0
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
for i in range(pred_y.shape[0]):
    if test_y[i] ==pred_y[i]:
        s = s+1
torch.save(resnet.state_dict(),'resnet18.pth')
print(s/pred_y.shape[0], 'resnet prediction accuary')
print('resnet fp32time',end-start)

#fc剪枝，使用pytorch自带工具
m = resnet.fc.weight
parameter_renet=(
    (resnet.conv1,'weight'),
    (resnet.fc,'weight')
)
prune.global_unstructured(
    parameter_renet,
    pruning_method=prune.L1Unstructured,
    amount=0.2
)
start = time.time()
test_output = resnet(test_x)
end = time.time()
s = 0
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
for i in range(pred_y.shape[0]):
    if test_y[i] ==pred_y[i]:
        s = s+1
print(s/pred_y.shape[0], 'resnet prediction accuary')
print('resnet fp32time',end-start)
print(m-resnet.fc.weight)

resnet.eval()
qconfig = torch.quantization.get_default_qconfig('fbgemm')
qconfig_dict = {"":qconfig}
def calibrate(model,data_loader):
    model.eval()
    with torch.no_grad():
        for image,target in data_loader:
            model(image)
prepare_model = prepare_fx(resnet,qconfig_dict)
calibrate(prepare_model,train_loader)
quantized_model = convert_fx(prepare_model)
start = time.time()
test_outputq = quantized_model(test_x)
end = time.time()
pred_y = torch.max(test_outputq, 1)[1].data.numpy().squeeze()
s=0
for i in range(pred_y.shape[0]):
    if test_y[i] ==pred_y[i]:
        s = s+1
print(s/pred_y.shape[0],'fx predication accuary')
print('int8time',end-start)
input_size = (1,28,28)
x = torch.randn(BATCH_SIZE,*input_size)
torch.onnx.export(quantized_model,torch.randn(BATCH_SIZE,1,28,28),'resnet18_fx.onnx')
#torch.onnx.export(resnet,torch.randn(BATCH_SIZE,1,28,28),'resnet18.onnx')