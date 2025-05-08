
import numpy as np
import pandas as pd
from torch.optim import AdamW
from uitls_8822_capacity import computation_time
import math
from time import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from torch.cuda.amp import GradScaler, autocast

from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d,ReLU6,GELU


from numpy import mat

x = pd.read_csv(r'/home/wwj/chenqiliang/8822_list.csv',header=None)
matrix = x.values



def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i, 4]


a=1
Nt = 2
P_t =0.01
P_A =0.35
N_s =2
P_ct =0.0344 
N_r =2
P_cr =0.0625
P_sync = 0.050
B_W = 10000000
dataset = pd.read_csv(r'/home/wwj/chenqiliang/SVD/All-channel_matrix_p_1.csv').iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)
label = pd.read_csv(r'/home/wwj/chenqiliang/SVD/capacity_labels_p_1.csv').iloc[:, 1]
label = np.asarray(label, np.int32)
label.astype(np.int32)
#one hot
n_class = 784
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1



xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.2, random_state=40)
print("xTrain: ", len(xTrain))
print(xTrain.shape)

print("xTest: ", len(xTest))






#
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv3(x)
        out3 = self.conv5(x)
        return out1 + out2 + out3  # 

# 
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
       
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
        
         # 
        self.multi_scale_block = MultiScaleBlock(out_planes)  # 

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        #
       
        out = self.multi_scale_block(out)  # 
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)  # 

# 
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

# 
class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=10, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=2, stride=1, padding=1, bias=False)
        
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                # 
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:  # 
                    m.bias.data.zero_()

    def forward(self, x):
        x = x.reshape(-1, 1, 8, 8)  # 
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        
        out = F.avg_pool2d(out, out.size()[2:])  # 
        out = out.view(-1, self.nChannels)  # 
        return self.fc(out)  #




train_dataset = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
test_dataset = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = WideResNet(784).to(device) 
#model.load_state_dict(torch.load(f"/home/wwj/chenqiliang/model.pth"))




lr = 0.001 
initial_lr=0.1
weight_decay = 0.0001  
betas = (0.95, 0.999)  
#eps = 1e-8
eps=1e-8
momentum=0.9
alpha=0.99
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)




criterion = nn.CrossEntropyLoss()
current_lr = optimizer.param_groups[0]['lr']
best_loss = float('inf') 


trainStart = time()
num_epochs = 30


 
for epoch in range(num_epochs):
    model.train()
    t = tqdm(train_loader, total=len(train_loader))

    for inputs, labels in t:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = torch.argmax(labels, dim=1)
        

        optimizer.zero_grad()
     

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
    if loss < best_loss:
        best_loss = loss.item()
        
        save_file_name = f"/home/wwj/chenqiliang/model.pth"  
        torch.save(model.state_dict(), save_file_name)

    print("epoch is {},loss is {}".format(epoch, loss))

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()




train = time() - trainStart



model = WideResNet(784)
model.load_state_dict(torch.load(f"/home/wwj/chenqiliang/model.pth"))
device = torch.device('cpu')
model.to(device)
ResNet_Pre1 = model(torch.from_numpy(xTest[0:40000]).to(device))


########################################################333wo xiede
testStart=time()
pre_array1 = np.zeros((40000, n_class))
index1=torch.argmax(ResNet_Pre1,dim=1)
for i in range(40000):
    pre_array1[i,index1[i]] = 1
    
    
test = time() - testStart   

aaa1 = torch.argmax(ResNet_Pre1, axis=1) + 1
ResNet_Pre_np1 = aaa1.numpy()
ResNet_Pre_np1_ = pre_array1
ResNet_Pre_np_=ResNet_Pre_np1_





############################################################3















###############################################################################




xTest_np = np.array(xTest[0:40000])


##################dui yu ce chuli

label_array = np.zeros((n_sample, n_class))
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1
###################
yTest_indices = np.array(yTest[:40000])





b=np.all(ResNet_Pre_np_ == yTest_indices, axis=1)

acc = np.sum(b) / 40000.0 * 100.0





'''
I1 = np.eye(8)
I2 = np.eye(2)

Pre_Loss = []

Pre_Capacity = []
for i in range(40000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(ResNet_Pre_np1[i])  
    Pre_sub = ArrayA[[i1, i2]][:, [j1, j2]]
    Pre_fullCapacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
    Pre_subCapacity= np.log2(np.linalg.det(I2 + a *  Pre_sub.T *  Pre_sub / 2))

    Pre_Capacity.append(Pre_subCapacity)
    Pre_Loss.append(Pre_fullCapacity - Pre_subCapacity)


Capacity_Mean = np.mean(Pre_Capacity)
Loss_Mean = np.mean(Pre_Loss)
Loss_Variance = np.var(Pre_Loss)


#print("acc is %.6f "%(acc))
print(f"{acc:.1f}%")
print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(Capacity_Mean)




print(Loss_Mean)
print(Loss_Variance)
'''









'''
I = np.eye(8)
I2 = np.eye(2)
Loss = []
Gain = []
for i in range(20000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(ResNet_Pre_np1[i])
    Pre_sub = mat(np.zeros((2, 2)), dtype=float)
    Pre_sub[0, 0] = ArrayA[i1, j1]
    Pre_sub[0, 1] = ArrayA[i1, j2]
    Pre_sub[1, 0] = ArrayA[i2, j1]
    Pre_sub[1, 1] = ArrayA[i2, j2]
    Pre_fullGian = math.sqrt(1 / 2) * np.linalg.norm(ArrayA, ord='fro')
    Pre_subGian = math.sqrt(1 / 2) * np.linalg.norm(Pre_sub, ord='fro')
    Gain.append(Pre_subGian)
    Loss.append(Pre_fullGian - Pre_subGian)

Gain_Mean = np.mean(Gain)
Loss_Mean = np.mean(Loss)
Loss_Variance = np.var(Loss)

print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(Gain_Mean)
print(Loss_Mean)
print(Loss_Variance)

print(f"{acc:.1f}%")
'''







I1 = np.eye(8)
I2 = np.eye(2)

Pre_Loss = []
Pre_EE = []
Pre_Capacity = []
for i in range(40000):
    ArrayA = xTest_np[i].reshape(8, 8)
    ArrayA = np.matrix(ArrayA)

    i1, i2, j1, j2 = GetIndexFrom(ResNet_Pre_np1[i])  
    Pre_sub = ArrayA[[i1, i2]][:, [j1, j2]]
    Pre_fullCapacity = np.log2(np.linalg.det(I1 + a * ArrayA.T * ArrayA / 8))
    Pre_subCapacity= np.log2(np.linalg.det(I2 + a *  Pre_sub.T *  Pre_sub / 2))


    P_total = (P_t / P_A) + (N_s * P_ct) + (N_r * P_cr) + 2 * P_sync
    EE=(B_W * Pre_subCapacity) / P_total

    Pre_EE.append(EE)
    EE_Mean = np.mean(Pre_EE)
















print(f"{acc:.6f}%")
print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(EE_Mean)
#print(Loss_Mean)
#print(Loss_Variance)









