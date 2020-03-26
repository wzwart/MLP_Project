import torch
import numpy as np
from torch.nn.utils import prune
import torch.nn as nn
import pandas as pd
class Pruner():

    def __init__(self, layer_dict, prune_prob, pruning_method=None, top_pruner=True):
        self.layer_dict=layer_dict
        self.prune_prob=prune_prob
        self.top_pruner=top_pruner
        if pruning_method is None:
            self.pruning_method = "adhoc"
        else:
            self.pruning_method = pruning_method
        if prune_prob>0:
            self.init_pruning_masks()

    def init_pruning_masks(self):
        self.pruning_masks={}
        self.weight_count=0
        self.weight_count_pruned = 0
        self.parameters_to_prune=[]
        for layer in self.layer_dict:
            if hasattr(self.layer_dict[layer], 'weight'):
                self.weight_count+=self.layer_dict[layer].weight.numel()
                self.parameters_to_prune.append((self.layer_dict[layer], "weight"))
                if self.pruning_method=="random_unstructured":
                    print(layer, self.layer_dict[layer])
                    if isinstance(self.layer_dict[layer], nn.Conv2d):
                        prune.random_unstructured(self.layer_dict[layer], name="weight", amount=self.prune_prob)
                        self.weight_count_pruned += int(self.layer_dict[layer].weight.numel()*self.prune_prob)
                    elif isinstance(self.layer_dict[layer], nn.ConvTranspose2d):
                            prune.random_unstructured(self.layer_dict[layer], name="weight", amount=self.prune_prob)
                            self.weight_count_pruned += int(self.layer_dict[layer].weight.numel() * self.prune_prob)
                    else:
                        self.weight_count_pruned += 0
                elif self.pruning_method=="adhoc":
                    self.pruning_masks[str(layer)]=np.random.random_sample((self.layer_dict[layer].weight.shape))>self.prune_prob
                    self.weight_count_pruned += int(self.layer_dict[layer].weight.numel() * self.prune_prob)
            if hasattr(self.layer_dict[layer], 'layer_dict'):
                self.layer_dict[layer].pruner = Pruner(self.layer_dict[layer].layer_dict, self.prune_prob, pruning_method=self.pruning_method, top_pruner=False)
                self.weight_count+= self.layer_dict[layer].pruner.weight_count
                self.weight_count_pruned += self.layer_dict[layer].pruner.weight_count_pruned
                self.parameters_to_prune = self.parameters_to_prune + self.layer_dict[layer].pruner.parameters_to_prune
        if self.pruning_method=="global":
            prune.global_unstructured(
                tuple(self.parameters_to_prune),
                pruning_method=prune.L1Unstructured,
                amount=self.prune_prob,
            )
            data = []

            for parameter_to_prune in self.parameters_to_prune:
                print(sum(p.numel() for p in parameter_to_prune[0].parameters() if p.requires_grad))
                data_temp = [parameter_to_prune[0],100. * float(torch.sum(parameter_to_prune[0].weight == 0))/ float(parameter_to_prune[0].weight.nelement()),sum(p.numel() for p in parameter_to_prune[0].parameters() if p.requires_grad)]
                data.append(data_temp)
                #print(
                    #f"Sparsity in { parameter_to_prune[0]}: {100. * float(torch.sum(parameter_to_prune[0].weight == 0))/ float(parameter_to_prune[0].weight.nelement()):.2f}%"
                #)

            df = pd.DataFrame(data, columns=['Layer', 'Pruned Percentage','Weights'])
            direct = 'exp_jan_prune/example_jan' +'_'+str(self.prune_prob)+'.csv'
            df.to_csv(direct,index=False)
            data2 = pd.read_csv(direct)
            new = data2["Layer"].str.split("(", n=1, expand=True)
            data2.drop(columns=["Layer"], inplace=True)
            data2["Layer"] = new[0]
            columns_titles = ["Layer", "Pruned Percentage",'Weights']
            data2 = data2.reindex(columns=columns_titles)
            data2.to_csv(direct, index=False)
    def prune(self, device):
        if self.prune_prob==0 or self.pruning_method !="adhoc": # return quickly
            return
        for layer in self.layer_dict:
            if hasattr(self.layer_dict[layer], 'weight'):
                with torch.no_grad():
                    self.layer_dict[layer].weight*=torch.Tensor(self.pruning_masks[str(layer)]).to(device=device)
            if hasattr(self.layer_dict[layer], 'pruner'):
                self.layer_dict[layer].pruner.prune(device=device)

