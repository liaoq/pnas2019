import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import math

device = torch.device("cuda:2")
print(device)
torch.cuda.get_device_name(device)

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


# In[3]:


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, drop_last=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

sampletrainloader=torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False, num_workers=4)

sampletestloader=torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=4)

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# In[ ]:


def scale(m, scaling_factor):
    if type(m) == nn.Linear:
        m.weight.data = m.weight.data * scaling_factor
        m.bias.data = m.bias.data * scaling_factor
    if type(m) == nn.Conv2d:
        m.weight.data = m.weight.data * scaling_factor
        
def standardDeviationVersusLossSGD(std_dev_list, learning_rates, num_epochs, trainloader, testloader, classes, device):
    colors = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'w-']
    legend = []
    histories = []
    with open('new_sample_data.pkl', 'rb') as input:
        sample_data = pickle.load(input)    
        for index, scaling_factor in enumerate(std_dev_list):
            torch.cuda.empty_cache()
            net = resnet56_cifar()
            net = net.to(device)
            net.apply(lambda x:scale(x, scaling_factor))
            print("Scaling factor: ", scaling_factor)
            lossFn = nn.CrossEntropyLoss()

            def num_correct (outputs, labels):
                n = len(outputs)
                count = 0
                for i in range (n):
                    if outputs[i].max(0)[1] == labels[i]:
                        count += 1
                return count


            history = {}
            history["scaling_factor"] = scaling_factor
            history["epoch"] = []
            history["training_loss"] = []
            history["training_accuracy"] = []
            history["testing_loss"] = []
            history["testing_accuracy"] = []
            iter = 0
            for learning_rate in learning_rates:
                optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
                for epoch in range(iter * num_epochs, (iter+1)*num_epochs):
                    net.train()
                    history["epoch"].append(epoch)
                    print("Epoch: ", epoch)
                    running_loss = 0
                    running_num = 0
                    running_count = 0
                    i = 0
                    for data, labels in trainloader:
                        data, labels = data.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = net(data)
                        loss = lossFn(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_num += num_correct(outputs, labels)
                        running_loss += loss.item()
                        running_count += 128
                        i+=1

                    print ("training loss:", running_loss/i)
                    history["training_loss"].append(running_loss/i)
                    print ("training accuracy:", running_num/running_count)
                    history["training_accuracy"].append(running_num/running_count)

                    net.eval()
                    running_loss = 0
                    running_num = 0
                    running_count = 0
                    i = 0

                    with torch.no_grad():
                        for data, labels in testloader:
                            data, labels = data.to(device), labels.to(device)
                            outputs = net(data)
                            loss = lossFn(outputs, labels)
                            running_num += num_correct(outputs, labels)
                            running_loss += loss.item()
                            running_count += 100
                            i+=1
                    print ("testing loss:", running_loss/i)
                    history["testing_loss"].append(running_loss/i)
                    print ("testing accuracy: ", running_num/running_count)
                    history["testing_accuracy"].append(running_num/running_count)
                    torch.save(net.state_dict(), 'net3.pt')
                iter += 1

            net.eval()

            sample_output = net(sample_data.to(device))
            print("sample output", sample_output)
            sample_norm = torch.norm(sample_output)


            running_loss = 0
            running_num = 0
            with torch.no_grad():
                for data, labels in sampletestloader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = net(data)
                    outputs2 = outputs/sample_norm
                    loss = lossFn(outputs2, labels)
                    running_num += num_correct(outputs2, labels)
                    running_loss += loss.item()
            print ("normalized testing loss:", running_loss/50)
            history["normalized_testing_loss"] = running_loss/50
            print ("normalized testing accuracy: ", running_num/10000)
            history["normalized_testing_accuracy"] = running_num/10000

            running_loss = 0
            running_num = 0
            with torch.no_grad():
                for data, labels in sampletrainloader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = net(data)
                    outputs2 = outputs/sample_norm
                    loss = lossFn(outputs2, labels)
                    running_num += num_correct(outputs2, labels)
                    running_loss += loss.item()
            print ("normalized training loss:", running_loss/250)
            history["normalized_training_loss"] = running_loss/250
            print ("normalized training accuracy: ", running_num/50000)
            history["normalized_training_accuracy"] = running_num/50000

            histories.append(history)
            save_object(histories, 'newhistories3.pkl')
    return histories

histories1 = standardDeviationVersusLossSGD(std_dev_list = [.75, .8],
                               learning_rates=[0.01, 0.001, 0.0001], 
                               num_epochs=100,
                               trainloader=trainloader,
                               testloader=testloader,
                               classes=classes,
                               device = device)
