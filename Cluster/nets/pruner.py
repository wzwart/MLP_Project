import torch
import numpy as np


class Pruner():
    def __init__(self, layer_dict, prune_prob):
        self.layer_dict=layer_dict
        self.prune_prob=prune_prob
        if prune_prob>0:
            self.init_pruning_masks()

    def init_pruning_masks(self):
        self.pruning_masks={}
        for layer in self.layer_dict:
            print(layer)
            if hasattr(self.layer_dict[layer], 'weight'):
                self.pruning_masks[str(layer)]=np.random.random_sample((self.layer_dict[layer].weight.shape))<self.prune_prob
            if hasattr(self.layer_dict[layer], 'layer_dict'):
                self.layer_dict[layer].pruner = Pruner(self.layer_dict[layer].layer_dict, self.prune_prob)

    def prune(self, device):
        if self.prune_prob==0: # return quickly
            return
        for layer in self.layer_dict:
            if hasattr(self.layer_dict[layer], 'weight'):
                with torch.no_grad():
                    self.layer_dict[layer].weight*=torch.Tensor(self.pruning_masks[str(layer)]).to(device=device)
            if hasattr(self.layer_dict[layer], 'pruner'):
                self.layer_dict[layer].pruner.prune(device=device)

