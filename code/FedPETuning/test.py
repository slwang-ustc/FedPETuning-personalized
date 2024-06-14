"""Test some code snippets"""
# bash fed_run.sh /data/slwang/FedPETuning_personalized rte fedavg 10002 0 1 2 3 4 5

import torch
import torch.nn as nn
from torch.nn import functional as F

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

for idx, (name, params) in enumerate(backbone.named_parameters()):
    print(name, idx)

Visualization(backbone).structure_graph()
