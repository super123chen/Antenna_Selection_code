from __future__ import division, print_function, absolute_import
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
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d,ReLU6,GELU

n_class = 784  
signal_power = 1.0 

x = pd.read_csv(r'/home/wwj/chenqiliang/8822_list.csv',header=None)
matrix = x.values



def GetIndexFrom(y_pre):
    for i in range(0, 784):
        if y_pre == matrix[i][0]:
            return matrix[i, 1], matrix[i, 2], matrix[i, 3], matrix[i, 4]


a = 10
dataset = pd.read_csv(r'/home/wwj/chenqiliang/wubiaoqian/All-channel_matrix_p_20.csv').iloc[:, 1:]
dataset = np.asarray(dataset, np.float32)
dataset = dataset.reshape(dataset.shape[0], 8, 8, 1)
label = pd.read_csv(r'/home/wwj/chenqiliang/wubiaoqian/capacity_labels_p_20.csv').iloc[:, 1]
label = np.asarray(label, np.int32)
label.astype(np.int32)

#one hot
n_class = 784
n_sample = label.shape[0]
label_array = np.zeros((n_sample, n_class))
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1



xTrain, xTest, yTrain, yTest = train_test_split(dataset, label_array, test_size=0.1, random_state=40)
print("xTrain: ", len(xTrain))
print(xTrain.shape)

print("xTest: ", len(xTest))



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
    
    
    
    
    
    
    
    
class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
    
    



class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out       
######
'''
class Shadow(nn.Module):
    def __init__(self, inc):
        super(Shadow, self).__init__()
        self.lin1 = nn.Linear(int(inc / 4), int(inc / 4))
        self.lin2 = nn.Linear(int(inc / 4), int(inc / 4))
        self.lin3 = nn.Linear(int(inc / 4), int(inc / 4))
        self.lin4 = nn.Linear(int(inc / 4), int(inc / 4))
        self.conv = nn.Conv2d(int(2 * inc), inc, 1)
        

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # bhwc
        x_chunks = torch.chunk(x, chunks=4, dim=3)
        x_chunk1, x_chunk2, x_chunk3, x_chunk4 = x_chunks

        x_chunk1 = self.lin1(x_chunk1)
        x_chunk1 = F.relu(x_chunk1)
        x_chunk2 = self.lin2(x_chunk2)
        x_chunk2 = F.gelu(x_chunk2)
        x_chunk3 = self.lin3(x_chunk3)
        x_chunk3 = F.selu(x_chunk3)
        x_chunk4 = self.lin4(x_chunk4)
        x_chunk4 = x_chunk4*F.sigmoid(x_chunk4)
        

        x = torch.cat([x, x_chunk1, x_chunk2, x_chunk3, x_chunk4], dim=3)
        
       

        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)

       
        return x   
'''







class Resnet(nn.Module):
    def __init__(self, n_class):
        super(Resnet, self).__init__()
        self.model0 = Sequential(
           
            
            Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            #Shadow(64),
            ReLU(),
          
        )
        self.PSA = PSAModule(64,64)
        
        self.R0 = ReLU()
        
        self.model1 = Sequential(
            
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            #Shadow(64),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            
            ReLU(),
        )
        self.PSA = PSAModule(64,64)
        self.R1 = ReLU()
       
        self.model2 = Sequential(
            
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            #Shadow(64),
            ReLU(),
            GELU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            #Shadow(64),
            ReLU(),
            
        )
        self.PSA = PSAModule(64,64)
        self.R2 = ReLU()

        self.model3 = Sequential(
           
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            #Shadow(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            #Shadow(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            #Shadow(128),
            ReLU(),
        )
        
        self.R3 = ReLU()

        self.model4 = Sequential(
           
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            #Shadow(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            #Shadow(128),
            ReLU(),
        )
        
        self.R4 = ReLU()

        self.model5 = Sequential(
           
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            #Shadow(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            #Shadow(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            #Shadow(256),
            ReLU(),
        )
        
        self.R5 = ReLU()
        

        self.model6 = Sequential(
            
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            #Shadow(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            #Shadow(256),
            ReLU(),
        )
        
        self.R6 = ReLU()

        self.model7 = Sequential(
           
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            #Shadow(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            #Shadow(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            #Shadow(512),
            ReLU(),
        )
        
        self.R7 = ReLU()

        self.model8 = Sequential(
            
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            #Shadow(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            #Shadow(512),
            ReLU(),
        )
       
        self.R8 = ReLU()

        
        self.aap = AdaptiveAvgPool2d((1, 1))
       
        self.flatten = Flatten(start_dim=1)
        
        self.fc = Linear(512, n_class)

    def forward(self, x):
        x = x.reshape(-1, 1, 8, 8)
        x = self.model0(x)
        #x = self.PSA(x)
        
        f1 = x
        x = self.model1(x)
        x = x + f1
        x = self.R1(x)
        

        f1_1 = x
        x = self.model2(x)
        x = x + f1_1
        x = self.R2(x)
        

        f2_1 = x
        f2_1 = self.en1(f2_1)
        x = self.model3(x)
        x = x + f2_1
        x = self.R3(x)
        

        f2_2 = x
        x = self.model4(x)
        x = x + f2_2
        x = self.R4(x)
        

        f3_1 = x
        f3_1 = self.en2(f3_1)
        x = self.model5(x)
        x = x + f3_1
        x = self.R5(x)
        

        f3_2 = x
        x = self.model6(x)
        x = x + f3_2
        x = self.R6(x)
        

        f4_1 = x
        f4_1 = self.en3(f4_1)
        x = self.model7(x)
        x = x + f4_1
        x = self.R7(x)
        

        f4_2 = x
        x = self.model8(x)
        x = x + f4_2
        x = self.R8(x)
        

        
        x = self.aap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x








train_dataset = TensorDataset(torch.Tensor(xTrain), torch.Tensor(yTrain))
test_dataset = TensorDataset(torch.Tensor(xTest), torch.Tensor(yTest))

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Resnet(784).to(device)
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
num_epochs = 40



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


train = time() - trainStart












model = Resnet(784)
model.load_state_dict(torch.load(f"/home/wwj/chenqiliang/model.pth"))
device = torch.device('cpu')
model.to(device)

ResNet_Pre1 = model(torch.from_numpy(xTest[0:20000]).to(device))


########################################################333wo xiede
testStart=time()
pre_array1 = np.zeros((20000, n_class))
index1=torch.argmax(ResNet_Pre1,dim=1)
for i in range(20000):
    pre_array1[i,index1[i]] = 1
    
    
test = time() - testStart   

aaa1 = torch.argmax(ResNet_Pre1, axis=1) + 1
ResNet_Pre_np1 = aaa1.numpy()
ResNet_Pre_np1_ = pre_array1
ResNet_Pre_np_=ResNet_Pre_np1_





############################################################3















###############################################################################




xTest_np = np.array(xTest[0:20000])


##################dui yu ce chuli

label_array = np.zeros((n_sample, n_class))
for i in range(n_sample):
    label_array[i, label[i] - 1] = 1
###################
yTest_indices = np.array(yTest[:20000])





b=np.all(ResNet_Pre_np_ == yTest_indices, axis=1)

acc = np.sum(b) / 20000.0 * 100.0



ResNet_PredictedIndices = np.argmax(ResNet_Pre_np_, axis=1)







I1 = np.eye(8)
I2 = np.eye(2)

Pre_Loss = []

Pre_Capacity = []
for i in range(20000):
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
print(f"{acc:.2f}%")
print("160000traintime%.1f %s" % (computation_time(train)[0], computation_time(train)[1]))
print("40000testtime%.1f %s" % (computation_time(test)[0], computation_time(test)[1]))

print(Capacity_Mean)
print(Loss_Mean)
print(Loss_Variance)




predicted_classes = torch.argmax(ResNet_Pre1, dim=1).numpy()  
true_classes = np.argmax(yTest[:20000], axis=1) 


def calculate_ber(predicted, true):
   
    predicted_binary = np.zeros((predicted.size, n_class), dtype=bool) 
    true_binary = np.zeros((true.size, n_class), dtype=bool)

    for i in range(predicted.size):
        predicted_binary[i, predicted[i]] = 1
        true_binary[i, true[i]] = 1

   
    errors = np.sum(predicted_binary != true_binary)

   
    total_bits = predicted_binary.size

   
    ber = errors / total_bits
    return ber

ber = calculate_ber(predicted_classes, true_classes)

print(f"Bit Error Rate (BER): {ber:.6f}")



























