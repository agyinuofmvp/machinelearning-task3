# 这段代码存在问题，一开始我想的是本身CIFAR-10数据集就不大，就想着用预训练的权重，但是一直显示我无法加载
# 网络连接出现问题，但是我又检查了网络，发现可以访问，后面我又试着不使用预权重，但是程序很久很久都没有结果
# 但是截止时间要到了，我就把这段有问题的代码上传了
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import timm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


num_epochs = 10
batch_size = 64
# 加载数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小,往下看就知道为什么调整了
    transforms.ToTensor()
])
trainset = CIFAR10(root='./data1', train=True, download=True, transform=transform)
testset = CIFAR10(root='./data1', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# 使用timm库直接加载ViT模型
#划分成16x16，输入图像大小：224x224，pretrained需不需要使用在大规模数据集上预训练的权重，num_class输出类别
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # 提取预测的类别，1指定了维度，表示在每一行寻找最大值及其对应的索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

if __name__ == "__main__":
    train()
    test()