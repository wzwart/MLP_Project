import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from arg_extractor import get_args
args, device = get_args()  # get arguments from command line
currentDirectory = os.getcwd()
summary = str(currentDirectory) + '/'+args.experiment_name_graph[0]+'/result_outputs/summary.csv'
df = pd.read_csv(summary)
fig = px.line(df)

for i in range(0,len(args.experiment_name_graph),1):
    summary = str(currentDirectory) + '/' + args.experiment_name_graph[i] + '/result_outputs/summary.csv'
    df = pd.read_csv(summary)
    fig.add_trace(go.Scatter(x=df['curr_epoch'], y=df['train_loss'],
                        mode='lines',
                        name='train_loss' +' '+ args.experiment_name_graph[i]))
    fig.add_trace(go.Scatter(x=df['curr_epoch'], y=df['val_loss'],
                        mode='lines',
                        name='val_loss'+' '+ args.experiment_name_graph[i]))
fig.update_layout(
    xaxis_title="Epoch",
    yaxis_title="Loss",yaxis_type="log",yaxis = dict(

        dtick = 1
    ),legend=dict(
        x=0.5,
        y=1,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    )
)
graph = str(currentDirectory) + '/graphs_multiple_experiments'
if not os.path.exists(graph):
    os.mkdir(graph)
fig.write_image(graph + '/'+str(args.experiment_name_graph) +'.png')
