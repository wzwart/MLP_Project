import glob
import os
import json
import pandas as pd
import numpy as np

df = pd.DataFrame()
dirs=[]
patterns=["exp_*","Exp*","EXP*","base_*","Base_*","BASE_*"]
dirs = dirs+ glob.glob("exp_*")
dirs=[dir for dir in dirs if os.path.isdir(dir)]
dirs=list(set(dirs))
for dir in dirs:
    json_files = glob.glob(dir+"/*.json")
    if len(json_files) > 1:
        print("there are more than 1 json files")
    if len(json_files)>=1:
        with open(json_files[0]) as f:
            data = json.load(f)
    else:
        data ={}
    data['name']=dir

    train_results_file= os.path.join(dir,"result_outputs/summary.csv")
    if os.path.isfile(train_results_file):
        train_results=pd.read_csv(train_results_file)
        columns=train_results.columns
        for parameter in ["train_nme", "val_nme", "train_acc", "val_acc", "train_loss", "val_loss"]:
            if parameter in columns:
                min=train_results[parameter].min()
                idx_min=train_results[parameter].idxmin()
                data[parameter+"_min"] = min
                data[parameter+"_min_epoch"] = idx_min

    train_results_file= os.path.join(dir,"result_outputs/test_summary.csv")
    if os.path.isfile(train_results_file):
        train_results=pd.read_csv(train_results_file)
        columns=train_results.columns
        for parameter in ["test_nme", "test_acc", "test_loss"]:
            if parameter in columns:
                val=train_results[parameter].values[0]
                data[parameter] = val
    df=df.append(data, ignore_index=True)

df = df.replace(np.nan, '', regex=True)
df.set_index('name', inplace=True)



df= df.transpose()

for dir in dirs:
    df=df.rename(columns={dir: dir.replace("exp_","").replace("_", " ")})

df=df.drop(["experiment_name_graph","filepath_to_data_1","filepath_to_data_2","experiment_name"])

df.to_excel("results.xlsx")

for dir in dirs:
    df = df.rename(columns={dir: dir.replace(" ", "\n")})




def hover(hover_color="#ffff99"):
    return dict(selector="tr:hover",
                props=[("background-color", "%s" % hover_color)])



styles = [
    hover(),
    dict(selector="th", props=[("text-align", "center")]),
    dict(selector="caption", props=[("caption-side", "bottom")])
]

html = (
    df.style.set_table_styles(styles)
    .render()
)

f=open("results.html", "w")
f.write(html)
f.close()

print (df)