# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import copy


class Aggregators(object):
    """Define the algorithm of parameters aggregation"""

    @staticmethod
    def fedavg_aggregate(serialized_params_list, weights=None):
        """FedAvg aggregator

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Merge all tensors following FedAvg.
            weights (list, numpy.array or torch.Tensor, optional): Weights for each params, the length of weights need to be same as length of ``serialized_params_list``

        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        weights = weights / torch.sum(weights)
        assert torch.all(weights >= 0), "weights should be non-negative values"
        serialized_parameters = torch.sum(
            torch.stack(serialized_params_list, dim=-1) * weights, dim=-1)

        return serialized_parameters


    @staticmethod
    def persona_aggregate(
        serialized_params_list, global_params, 
        client_params_idxes, model
    ):
        """FedAvg aggregator

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Merge all tensors following FedAvg.
            weights (list, numpy.array or torch.Tensor, optional): Weights for each params, the length of weights need to be same as length of ``serialized_params_list``

        Returns:
            torch.Tensor
        """
        # if weights is None:
        #     weights = torch.ones(len(serialized_params_list))

        # if not isinstance(weights, torch.Tensor):
        #     weights = torch.tensor(weights)

        # weights = weights / torch.sum(weights)
        # assert torch.all(weights >= 0), "weights should be non-negative values"
        # serialized_parameters = torch.sum(
        #     torch.stack(serialized_params_list, dim=-1) * weights, dim=-1
        # )

        # return serialized_parameters

        global_agg_num = {}  # key: params_idx, value: number of clients

        for params_idxes in client_params_idxes:
            for params_idx in params_idxes.int().tolist():
                if global_agg_num.get(params_idx) == None:
                    global_agg_num[params_idx] = 1
                else:
                    global_agg_num[params_idx] += 1

        global_params_temp = copy.deepcopy(global_params)
        for client_index, params_idxes in enumerate(client_params_idxes):
            params_idxes = params_idxes.int().tolist()
            current_index = 0  # keep track of where to read from grad_update
            for params_idx, parameter in enumerate(model.parameters()):
                numel = parameter.data.numel()
                if params_idx in params_idxes:
                    agg_weight = 1 / global_agg_num[params_idx]
                    global_params[current_index: current_index + numel] += (
                        (
                            serialized_params_list[client_index][current_index: current_index + numel] - global_params_temp[current_index: current_index + numel]
                        ) * agg_weight
                    )
                    # print('=====================================================', agg_weight)
                current_index += numel
        return global_params

    @staticmethod
    def fedasync_aggregate(server_param, new_param, alpha):
        """FedAsync aggregator
        
        Paper: https://arxiv.org/abs/1903.03934
        """
        serialized_parameters = torch.mul(1 - alpha, server_param) + \
                                torch.mul(alpha, new_param)
        return serialized_parameters
