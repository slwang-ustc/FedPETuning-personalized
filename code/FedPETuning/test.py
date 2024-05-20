"""Test some code snippets"""
# bash fed_run.sh /data/slwang/FedPETuning rte fedavg 10001 0 1 2 3 4 5

import torch
import torch.nn as nn
from torch.nn import functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
 
#         #三层全连接层
#         #wx+b
#         self.fc1 = nn.Linear(2, 4)
#         self.fc2 = nn.Linear(4, 2)
#         self.fc3 = nn.Linear(2, 5)
 
#     def forward(self, x):
#         # x: [b, 1, 28, 28]
#         x = F.relu(self.fc1(x)) #F.relu和torch.relu，用哪个都行
#         # x = F.relu(self.fc2(x))
#         # x = self.fc3(x)
 
#         return x

# parameters = []
# net = Net()
# for para in net.parameters():
#     # print(para.data.view(-1))
#     parameters.append(para.data.view(-1))
#     # print(para.grad)

# parameters = torch.cat(parameters)
# id_list = [torch.Tensor([1, 2, 3]).to(parameters.dtype)]
# print(parameters)
# print(id_list)
# content = id_list + [parameters]
# print(f'content: {content}')
# slice = [content[0].numel(), len(list(content[0].shape))] + list(content[0].shape)
# print(f'slice: {slice}')
# slices = [2, 2, 2]


x = torch.tensor([1,2,3,4])
y = torch.tensor([5,6,7,8])

para_list = [x, y]
weights = torch.ones(2)
weights = weights / torch.sum(weights)

para_list_stack = torch.stack(para_list, dim=-1)
print(torch.sum(para_list_stack * weights, dim=-1))

print(x, x.int())
