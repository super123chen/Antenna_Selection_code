
from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
from torch.optim import AdamW
#from uitls_8822_gain import computation_time
import math
from time import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
#import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch
import torchvision.models as models
# import torch
from ptflops import get_model_complexity_info


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
import matplotlib.pyplot as plt
from scipy.special import erfc  
from numpy import mat


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
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add a dimension for batching
        self.register_buffer("pe", pe)

    def forward(self, inputs):
        # Ensure inputs is shaped as (B, Seq, d_model)
        # Here, we should slice pe only for the second dimension
        # `inputs` must have 2nd dimension that matches the positional encoding
        size_seq = inputs.size(1)  # Get the sequence length
        pe_slice = self.pe[:, :size_seq]  # Get the matching positional encodings
        return self.dropout(inputs + pe_slice)


class Model(nn.Module):
    def __init__(self, inputs_size, outputs_size):  
        super(Model, self).__init__()
        self.dim_up = nn.Linear(inputs_size, 256)  
        self.positional_encoding = PositionalEncoding(256, 0.2)  
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=1)  
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1) 
        self.predict = nn.Linear(256, outputs_size)  
        self.activation = nn.PReLU() 
        self.shadow = Shadow(512) 
        self.dropout = nn.Dropout(0.2)  

    def transformer_encoder_forward(self, inputs):
        outputs = inputs.permute(1, 0, 2)  # (Seq, B, Feature)
        outputs = self.transformer_encoder(outputs)  # (Seq, B, Feature)
        outputs = outputs.permute(1, 0, 2)  # (B, Seq, Feature)
        outputs = outputs.mean(dim=1)  # (B, Feature)
        return outputs

    def forward(self, inputs):
        B = inputs.shape[0]
        inputs = inputs.view(B, -1)  # Flattening it to (B, inputs_size)

        outputs = self.dim_up(inputs)  # (B, 256)
        outputs = self.positional_encoding(outputs.unsqueeze(1))  # (B, Seq=1, 256)

        outputs = self.transformer_encoder_forward(outputs)  # (B, 256)
        outputs = self.activation(outputs)  # (B, 256)
        outputs = self.dropout(outputs)  # (B, 256)
        outputs = self.predict(outputs)  # (B, outputs_size)
        return outputs

# Create a model instance for testing
inputs = torch.randn(16, 1, 64)  # (B, Seq, Feature)
model = Model(inputs_size=64, outputs_size=784)
outputs = model(inputs)
print(outputs.shape)  # Should be (512, 784)
'''






class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, inputs):
        size_seq = inputs.size(1)
        pe_slice = self.pe[:, :size_seq]
        return self.dropout(inputs + pe_slice)

class Model(nn.Module):
    def __init__(self, inputs_size, outputs_size):
        super(Model, self).__init__()
        
        self.dim_up = nn.Linear(inputs_size, 64)
        self.dim_up = nn.Linear(inputs_size, 128)
        self.dim_up = nn.Linear(inputs_size, 256)
        self.dim_up = nn.Linear(inputs_size, 512)
        
         
        self.positional_encoding = PositionalEncoding(512, 0.2)
        # Increased number of heads and layers for better learning capacity
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=1)  
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.predict = nn.Linear(64, outputs_size)
        self.predict = nn.Linear(128, outputs_size)
        self.predict = nn.Linear(256, outputs_size) 
        self.predict = nn.Linear(512, outputs_size)
        self.activation = nn.Sigmoid()    # Changed to GELU
        #self.activation = nn.PReLU()
        self.shadow = Shadow(512)  
        self.dropout = nn.Dropout(0.2)
        
        
        
    
  
        
    def transformer_encoder_forward(self, inputs):
        outputs = inputs.permute(1, 0, 2)  # (Seq, B, Feature)
        outputs = self.transformer_encoder(outputs)  # (Seq, B, Feature)
        outputs = outputs.permute(1, 0, 2)  # (B, Seq, Feature)
        outputs = outputs.mean(dim=1)  # (B, Feature)
        return outputs

    def forward(self, inputs):
        B = inputs.shape[0]
        inputs = inputs.view(B, -1)  # Flattening it to (B, inputs_size)

        outputs = self.dim_up(inputs)  # (B, 256)
        outputs = self.positional_encoding(outputs.unsqueeze(1))  # (B, Seq=1, 256)

        outputs = self.transformer_encoder_forward(outputs)  # (B, 256)
        outputs = self.activation(outputs)  # (B, 256)
        outputs = self.dropout(outputs)  # (B, 256)
        outputs = self.predict(outputs)  # (B, outputs_size)
        return outputs

# Create a model instance for testing
inputs = torch.randn(16, 1, 64)  # (B, Seq, Feature)
model = Model(inputs_size=64, outputs_size=784)
outputs = model(inputs)
print(outputs.shape)  # Should be (16, 784)










###        
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








class Resnet(nn.Module):
    def __init__(self, n_class):
        super(Resnet, self).__init__()
        self.model0 = Sequential(
           
            
            Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            Shadow(64),
            ReLU(),
          
        )
        self.PSA = PSAModule(64,64)
        
        self.R0 = ReLU()
        
        self.model1 = Sequential(
            
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            Shadow(64),
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
            Shadow(64),
            ReLU(),
            GELU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(64),
            Shadow(64),
            ReLU(),
            
        )
        self.PSA = PSAModule(64,64)
        self.R2 = ReLU()

        self.model3 = Sequential(
           
            Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(128),
            Shadow(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            Shadow(128),
            ReLU(),
        )
        self.en1 = Sequential(
            Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(128),
            Shadow(128),
            ReLU(),
        )
        
        self.R3 = ReLU()

        self.model4 = Sequential(
           
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            Shadow(128),
            ReLU(),
            Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(128),
            Shadow(128),
            ReLU(),
        )
        
        self.R4 = ReLU()

        self.model5 = Sequential(
           
            Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(256),
            Shadow(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            Shadow(256),
            ReLU(),
        )
        self.en2 = Sequential(
            Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(256),
            Shadow(256),
            ReLU(),
        )
        
        self.R5 = ReLU()
        

        self.model6 = Sequential(
            
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            Shadow(256),
            ReLU(),
            Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(256),
            Shadow(256),
            ReLU(),
        )
        
        self.R6 = ReLU()

        self.model7 = Sequential(
           
            Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2, padding=1),
            BatchNorm2d(512),
            Shadow(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            Shadow(512),
            ReLU(),
        )
        self.en3 = Sequential(
            Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=2, padding=0),
            BatchNorm2d(512),
            Shadow(512),
            ReLU(),
        )
        
        self.R7 = ReLU()

        self.model8 = Sequential(
            
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            Shadow(512),
            ReLU(),
            Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            BatchNorm2d(512),
            Shadow(512),
            ReLU(),
        )
       
        self.R8 = ReLU()

        
        self.aap = AdaptiveAvgPool2d((1, 1))
       
        self.flatten = Flatten(start_dim=1)
        
        self.fc = Linear(512, n_class)

    def forward(self, x):
        x = x.reshape(-1, 1, 8, 8)
        x = self.model0(x)
        x = self.PSA(x)
        
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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Resnet(784).to(device)






















model_name = 'xxxx'
flops, params = get_model_complexity_info(model, (8, 8, 1), as_strings=True, print_per_layer_stat=True)
# print("%s |%s |%s" % (model_name, flops, params))
print("model_name",model_name)
print("flops:",flops)
print("params:",params)

