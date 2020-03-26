import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from arg_extractor import get_args
args, device = get_args()  # get arguments from command line
direct = 'exp_jan_prune/example_jan' +'_'+str(args.prune_prob)+'.csv'
data = pd.read_csv(direct)
#data = pd.read_csv(direct)
data.plot(kind='bar')
#plt.ylabel('Pruned Percentage')
#plt.xlabel('Layer')
plt.title(str(args.prune_prob))
plt.show()