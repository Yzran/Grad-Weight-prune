import torch
import torch.nn as nn
import time
import onnx
from torch.nn import functional as F
import os
import torch.utils.data as Data
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
from torchvision import datasets
from torch.quantization.quantize_fx import prepare_fx,convert_fx
import numpy as np
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD = False

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

class ResnetBasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(ResnetBasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,stride,1,bias=False)
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
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,stride[1],1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0,bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self,x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x+out)

class ResNet18_prunedable(nn.Module):
    def __init__(self,config = None):
        super(ResNet18_prunedable,self).__init__()
        if config is None:
            config = [64,64,64,64,64,128,128,128,128,128,256,256,256,256,256,512,512,512,512,512]
        self.conv1 = nn.Conv2d(3,config[0],kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(config[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = nn.Sequential(ResnetBasicBlock(config[0],config[2],1),
                                    ResnetBasicBlock(config[2],config[4],1))

        self.layer2 = nn.Sequential(
            ResnetdownBlock(config[4],config[7],[2,1]),
            ResnetBasicBlock(config[7],config[9],1)
        )
        self.layer3 = nn.Sequential(
            ResnetdownBlock(config[9], config[12], [2, 1]),
            ResnetBasicBlock(config[12], config[14], 1)
        )
        self.layer4 = nn.Sequential(
            ResnetdownBlock(config[14], config[17], [2, 1]),
            ResnetBasicBlock(config[17], config[19], 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(config[19],10)
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


def train(model,train_dataset,EPOCH,LR,prune):
    model.train()
    scale = 50000/BATCH_SIZE
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(train_dataset):
            x,y = x.cuda(),y.cuda()
            out = model(x)
            loss = loss_func(out,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            per = (step/scale) * 100
            now = '|' * int(per*0.5)
            total = ' ' * (49 - int(per *0.5))
            print("\rEPOCH:{} {:^3.0f}%[{}{}]".format(epoch,per, now, total), end='')
            #time.sleep(0.1)
        print('\t')
        test(model,test_loader)
    if prune == True:
        newmodel = grad_weight_prune(model,0.5,True)
        return newmodel

def test(model,test_dataset):
    model.eval()
    corr = 0
    for x,y in test_dataset:
        x, y = x.cuda(), y.cuda()
        out = model(x)
        pred = out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        corr += pred.eq(y.data.view_as(pred)).cpu().sum()
    acc = corr / len(test_loader.dataset) *100
    print('model accuracy is {}%'.format(acc))

def grad_weight_prune(model,prune_percent,cuda):

    config = []
    cfg_mask = []


    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            l1_list =[]
            mask = []
            for i in range(m.weight.shape[0]):
                weight_l1 = torch.norm(torch.flatten(m.weight[i]),p=1)
                grad_l1 = torch.norm(torch.flatten(m.weight.grad[i]),p=1)
                mixed_l1 = weight_l1*grad_l1
                l1_list.append(mixed_l1)
            l1_copy = l1_list.copy()
            l1_copy.sort()
            threshold = l1_copy[int(len(l1_copy)*prune_percent)]
            for i in range(m.weight.shape[0]):
                if l1_list[i] < threshold:
                    mask.append(0)
                else:
                    mask.append(1)
            mask = np.array(mask)
            mask = torch.tensor(mask)
            print(mask.shape[0])
            print(m.weight.data.shape)
            config.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))

    newmodel = ResNet18_prunedable(config=config)
    if cuda == True:
        newmodel.cuda()

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    print('new model parameters number is ',num_parameters)
    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    print(len(cfg_mask))
    conv_count = 0
    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            #在此处的模型中，bn层始终在卷积层之后，所以bn层后可以更新mask的数据

            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]

        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            # 根据mask对输入层的卷积核进行剪枝
            w1 = m0.weight.data[idx1.tolist(),:, :, :].clone()
            # 因为resnet特有的channel spilt层，所以对这一层进行特殊处理，需要保留上两侧的输入数量，这里只针对resnet18
            if conv_count ==5 or conv_count ==10 or conv_count ==15:
                tmplist = idx0.tolist()
            if conv_count ==7 or conv_count ==12 or conv_count ==17:
                w1 = w1[:, tmplist, :, :].clone()
            else:
                w1 = w1[:,idx0.tolist(),:,:]
            m1.weight.data = w1.clone()
            conv_count += 1
            continue

            #将新参数复制给newmodel
            m1.weight.data = m0.weight.data.clone()
    # 保存config数据，使得下次可以直接读取权重文件
    np.save('config{}.npy'.format(prune_percent),config)
    # 保存权重文件
    torch.save(newmodel.state_dict(),'resnet18_cifar10_{}pruned.pth'.format(prune_percent))
    return newmodel

resnet =ResNet18_prunedable()
resnet.cuda()
#train(resnet,train_loader,10,0.001)
#torch.save(resnet.state_dict(),'resnet18_cifar10.pth')
resnet.load_state_dict(torch.load('resnet18_cifar10.pth'))
# train(resnet,train_loader,1,LR,True)
saved_config = np.load('config0.5.npy')
print(saved_config)
resnet_pruned = ResNet18_prunedable(config=saved_config)
resnet_pruned.cuda()
resnet_pruned.load_state_dict(torch.load('resnet18_cifar10_0.5pruned.pth'))
train(resnet_pruned,train_loader,50,0.001,False)
torch.save(resnet_pruned.state_dict(),'resnet18_cifar10_0.5pruned.pth')