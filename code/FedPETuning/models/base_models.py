"""BaseModel for FedETuning"""

import copy
from abc import ABC
from utils import registry
from models.utils import PromptType

import torch
import torch.nn as nn
from transformers import trainer, AutoConfig

from opendelta import AutoDeltaConfig
from opendelta.auto_delta import AutoDeltaModel


class BaseModels(nn.Module, ABC):
    def __init__(self, task_name):
        super().__init__()

        self.task_name = task_name

        config = registry.get("config")
        self.model_config = config.model_config
        self.rank = config.federated_config.rank
        self.logger = registry.get("logger")

    def _build_config(self, **kwargs):
        auto_config = AutoConfig.from_pretrained(
            self.model_config.config_name if self.model_config.config_name else self.model_config.model_name_or_path,
            finetuning_task=self.task_name if self.task_name else None,
            # cache_dir=self.model_config.cache_dir,
            revision=self.model_config.model_revision,
            use_auth_token=True if self.model_config.use_auth_token else None,
            **kwargs
        )
        return auto_config

    def _build_model(self):
        backbone = self._add_base_model()

        if getattr(self.model_config, "permutation_layers", None):
            backbone = self._add_permutate_layers(backbone)

        if self.model_config.tuning_type:
            backbone = self._add_delta_model(backbone)

        return backbone

    def _add_base_model(self):
        raise NotImplementedError

    def _add_permutate_layers(self, backbone):
        # TODO only support BERT-NLU Task
        bert_modules = self.get_bert_module(backbone)
        old_modules = bert_modules.encoder.layer
        scrambled_modules = torch.nn.ModuleList()
        # Now iterate over all layers,
        # appending to the new module list according to the new order.
        if self.rank > 0:
            permutation = self.model_config.client_model_layers
        else:
            permutation = self.model_config.server_model_layers
        self.logger.debug(f"model's layer: {permutation}")
        for i in permutation:
            assert i <= 11, permutation
            scrambled_modules.append(old_modules[i])

        # Create a copy of the model, modify it with the new list, and return
        backbone_copy = copy.deepcopy(backbone)
        bert_modules_copy = self.get_bert_module(backbone_copy)
        bert_modules_copy.encoder.layer = scrambled_modules
        return backbone_copy

    def _add_delta_model(self, backbone):

        if any([True for PType in PromptType if PType in self.model_config.tuning_type]):
            # prefix tuning maybe in OpenDelta
            ...
        else:
            delta_args = registry.get("delta_config")
            delta_config = AutoDeltaConfig.from_dict(delta_args)
            delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=backbone)
            delta_model.freeze_module(set_state_dict=True)

            self.delta_type = delta_args['delta_type']
            # delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
            # self.logger.debug(delta_config)
            # self.logger.debug(backbone)
            # self.logger.debug(delta_args)

        return backbone
    
    # all
    def _map_name_with_layer_idx(self):
        name_idx_mapping = {}
        layer_idx = 0
        for name, _ in self.named_parameters():
            name_idx_mapping[name] = layer_idx
            layer_idx += 1
        return name_idx_mapping
    
    # [deltas + classifier], [LayerNorm]
    def _get_trainable_params_name(self):
        trainable_params_names = {
            "delta_trainable_params_names": [],
            "layer_norm_trainable_params_names": [],
            "classifier_trainable_params_names": [],
            "extra_embeddings_trainable_params_names": []
        }
        delta_count = 0
        layer_norm_count = 0
        layer_delta_trainable_params_names = []
        layer_classifier_trainable_params_names = []
        layer_norm_trainable_params_names = []
        layer_embedding_norm_trainable_params_names = []
        layer_extra_embeddings_trainable_params_names = []
        for name, _ in self.named_parameters():
            if self.delta_type in name:
                delta_count += 1
                layer_delta_trainable_params_names.append(name)
                if delta_count % 8 == 0:
                    trainable_params_names["delta_trainable_params_names"].append(layer_delta_trainable_params_names)
                    layer_delta_trainable_params_names = []
            elif "classifier" in name:
                layer_classifier_trainable_params_names.append(name)
            elif "LayerNorm" in name:
                if "embeddings.LayerNorm" in name:
                    layer_embedding_norm_trainable_params_names.append(name)
                else:
                    layer_norm_count += 1
                    layer_norm_trainable_params_names.append(name)
                    if layer_norm_count % 4 == 0:
                        trainable_params_names["layer_norm_trainable_params_names"].append(layer_norm_trainable_params_names)
                        layer_norm_trainable_params_names = []
            elif "extra_embeddings" in name:
                layer_extra_embeddings_trainable_params_names.append(name)
                
                
        trainable_params_names["classifier_trainable_params_names"].append(layer_classifier_trainable_params_names)
        trainable_params_names["layer_norm_trainable_params_names"].append(layer_embedding_norm_trainable_params_names)
        trainable_params_names["extra_embeddings_trainable_params_names"].append(layer_extra_embeddings_trainable_params_names)
                
        return trainable_params_names
    
    def _set_layernorms_trainable_params(self, model, tune_layernorms):
        for n, sub_module in model.named_modules():
            if isinstance(sub_module, nn.LayerNorm):
                for n, p in sub_module.named_parameters():
                    p.requires_grad = tune_layernorms

    def forward(self, inputs):
        raise NotImplementedError

    def get_bert_module(self, backbone):

        if self.model_config.model_type == "bert":
            return backbone.bert
        elif self.model_config.model_type == "roberta":
            return backbone.roberta
        elif self.model_config.model_type == "distilbert":
            return backbone.distilbert
        else:
            raise NotImplementedError
