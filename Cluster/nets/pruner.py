import torch
import numpy as np
from torch.nn.utils import prune

class Pruner():
    def __init__(self, layer_dict, prune_prob, pruning_method=None):
        self.layer_dict=layer_dict
        self.prune_prob=prune_prob
        if pruning_method is None:
            self.pruning_method = "adhoc"
        else:
            self.pruning_method = pruning_method
        if prune_prob>0:
            self.init_pruning_masks()

    def init_pruning_masks(self):
        self.pruning_masks={}
        for layer in self.layer_dict:
            print(layer)
            if hasattr(self.layer_dict[layer], 'weight'):
                if self.pruning_method=="random_unstructured":
                    prune.random_unstructured(self.layer_dict[layer], name="weight", amount=self.prune_prob)
                elif self.pruning_method=="adhoc":
                    self.pruning_masks[str(layer)]=np.random.random_sample((self.layer_dict[layer].weight.shape))<self.prune_prob
            if hasattr(self.layer_dict[layer], 'layer_dict'):
                self.layer_dict[layer].pruner = Pruner(self.layer_dict[layer].layer_dict, self.prune_prob, pruning_method=self.pruning_method)

    def prune(self, device):
        if self.prune_prob==0 or self.pruning_method !="adhoc": # return quickly
            return
        for layer in self.layer_dict:
            if hasattr(self.layer_dict[layer], 'weight'):
                with torch.no_grad():
                    self.layer_dict[layer].weight*=torch.Tensor(self.pruning_masks[str(layer)]).to(device=device)
            if hasattr(self.layer_dict[layer], 'pruner'):
                self.layer_dict[layer].pruner.prune(device=device)

