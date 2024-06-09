from collections import OrderedDict

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights, VGG16_Weights, AlexNet_Weights #added

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_Alex(nn.Module):
    def __init__(self, arch):
        super(VGG_Alex, self).__init__()
        # Model Selection
        num_classes = 10

        if arch.startswith('vgg16'):
            original_model=models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1) #modified because pretrained is deprecated, switched to weights
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )
            self.modelName = 'vgg16'
        elif arch.startswith('alexnet'):
            original_model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )
            self.modelName = 'alexnet'
        else:
            raise ValueError("Finetuning not supported on this architecture yet")

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        in_size = x.size(0)
        x = self.features(x)
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.classifier(x)
        return x


class Net(nn.Module): #resnet by original article
    def __init__(self, arch):
        super(Net, self).__init__()
        num_classes = 10
        # Model Selection
        original_model = models.__dict__[arch](weights=ResNet50_Weights.IMAGENET1K_V1)

        self.features = torch.nn.Sequential(
            *(list(original_model.children())[:-1])) 
        num_ftrs = original_model.fc.in_features
        self.classifier = nn.Linear(num_ftrs, num_classes)
        self.modelName = arch

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
        
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

#def ResNet18():
    #return ResNet(BasicBlock, [2, 2, 2, 2])

#def ResNet34():
    #return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

#def ResNet101():
    #return ResNet(Bottleneck, [3, 4, 23, 3])

#def ResNet152():
    #return ResNet(Bottleneck, [3, 8, 36, 3])

gradients = None
activations = None

model=ResNet50()

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]
    print(f'Gradients size: {gradients.size()}') 

def forward_hook(module, input, output):
    global activations
    activations = output
    print(f'Activations size: {activations.size()}')

# Register hooks for the last convolutional layer
last_conv_layer = model.layer4[-1].conv3
last_conv_layer.register_forward_hook(forward_hook)
last_conv_layer.register_full_backward_hook(backward_hook)

# Load model weights
model.load_state_dict(torch.load(".checkpoint/resnet50_cifar10_ckpt.pth", map_location=torch.device('cpu')))
model.eval()

if __name__ == '__main__':
    # Dummy arguments
    class args:
        def __init__(self):
            self.arch = "vgg16"

    # n = Net(None, args()) #Network on ImageNet (224 * 224 * 3)
    # print(n(torch.zeros(2, 3, 32, 32)).shape)

    model = {
        'alexnet': VGG_Alex('alexnet'),
        'vgg16': VGG_Alex('VGG16'),
        'resnet18': Net('resnet18'),
        'resnet50': Net('resnet50'),
    }[args.arch.lower()]

    # n = Net() #vgg16 on cifar10 (32 * 32 * 3)
    #n = Net('resnet18')  # resnet on cifar10 (32 * 32 * 3)
    # x = Variable(torch.FloatTensor(2, 3, 40, 40))
    #print(n(torch.zeros(2, 3, 32, 32)).shape)
    n = model[args.arch.lower()]
    print(n(torch.zeros(2,3,32,32)).shape)
