"""Test some code snippets"""
# bash fed_run.sh /data/slwang/FedPETuning_personalized rte fedavg 10001 0 1 2 3 4 5

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

from transformers import trainer, AutoConfig

from opendelta import AutoDeltaConfig
from opendelta.auto_delta import AutoDeltaModel
from bigmodelvis import Visualization

from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification

auto_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path = '/data/slwang/FedPETuning_personalized/pretrain/nlp/roberta-base/',
    finetuning_task='rte',
    # cache_dir=self.model_config.cache_dir,
    revision='main',
    use_auth_token=None,
    num_labels=2
)

backbone = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path='/data/slwang/FedPETuning_personalized/pretrain/nlp/roberta-base/',
    from_tf=False,
    config=auto_config,
    # cache_dir=self.model_config.cache_dir,
    revision='main',
    use_auth_token=None,
    # ignore_mismatched_sizes=self.model_config.ignore_mismatched_sizes,
)

delta_args = {'delta_type': 'adapter', 'learning_rate': 0.001, 'unfrozen_modules': ['deltas', 'layer_norm', 'final_layer_norm', 'classifier'], 'bottleneck_dim': 64}
delta_config = AutoDeltaConfig.from_dict(delta_args)
delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=backbone)


delta_model.freeze_module(set_state_dict=True)

# name_idx_mapping = {}
# layer_idx = 0
# for name, _ in backbone.named_parameters():
#     name_idx_mapping[name] = layer_idx
#     layer_idx += 1
# trainable_params_name = []
# layer_trainable_params_name = []
# count = 0
# for k, _ in backbone.state_dict().items():
#     if count % 8 == 0 and count != 0:
#         trainable_params_name.append(layer_trainable_params_name)
#         layer_trainable_params_name = []
#     layer_trainable_params_name.append(k)
#     count += 1
# trainable_params_name.append(layer_trainable_params_name)
# names = trainable_params_name[: 3]
# upload_params_idxes = []
# for layer_names in names:
#     for name in layer_names:
#         upload_params_idxes.append(name_idx_mapping[name])
# print(upload_params_idxes)

# delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
# Visualization(backbone).structure_graph()

