import numpy as np
import os
import getpass
import torch.nn as nn
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from adam16 import Adam16
from data_sets import data_providers as data_providers
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from nets.unet_dict import UNetDict
from data_sets.data_set_300_YT import Dataset_300W_YT


args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

import torch

torch.manual_seed(seed=args.seed)  # sets pytorch's seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
username = getpass.getuser()
username = str(username)
if args.dataset_name == '300W' or args.dataset_name == 'Youtube' or args.dataset_name == 'Both' or args.dataset_name == 'Benchmark':

    if os.path.isdir(args.filepath_to_data_1.replace("sxxxxxxx", username)):
        filepath_to_data = args.filepath_to_data_1.replace("sxxxxxxx", username)

        print(f"No1 {filepath_to_data} exists")

    elif os.path.isdir(args.filepath_to_data_2.replace("sxxxxxxx", username)):
        filepath_to_data = args.filepath_to_data_2.replace("sxxxxxxx", username)
        print(f"No2 {filepath_to_data} exists")

    else:

        raise FileExistsError

    try:
        max_size_dataset = int(args.max_size_dataset)
    except:
        max_size_dataset = -1
        pass

    width_in = 64
    height_in = 64
    width_out = 64
    height_out = 64
    if args.dataset_name == '300W':
        dataset_indicator = 0
    elif args.dataset_name == 'Youtube':
        dataset_indicator = 1
    elif args.dataset_name == 'Both':
        dataset_indicator = 2
    elif args.dataset_name == 'Benchmark':
        dataset_indicator = 3
    else:
        raise ValueError

    dataset = Dataset_300W_YT(
        root_dir=os.path.join(filepath_to_data),
        width_in=width_in, height_in=height_in, width_out=width_out,
        height_out=height_out,
        num_landmarks=args.num_landmarks,
        rbf_width=args.rbf_width,
        which_dataset=dataset_indicator,
        force_new_pickle=args.force_new_pickle,
        save_pickle=args.save_pickle,
        test_dataset=args.test_dataset,
        landmarks_collapsed=args.landmarks_collapsed,
        max_size=max_size_dataset,
        experiment=args.experiment_name)

    data_provider = data_providers.DataProviderFLD

    train_data = data_provider(dataset=dataset, which_set='train', batch_size=args.batch_size,
                               rng=rng)  # initialize our rngs using the argument set seed
    val_data = data_provider(dataset=dataset, which_set='valid', batch_size=args.batch_size,
                             rng=rng)  # initialize our rngs using the argument set seed
    test_data = data_provider(dataset=dataset, which_set='test', batch_size=args.batch_size,
                              rng=rng)  # initialize our rngs using the argument set seed

    data_provider=train_data

    if args.landmarks_collapsed:
        net = UNetDict(in_channel=3, out_channel=1, hour_glass_depth=args.Hourglass_depth, bottle_neck_channels=args.Hourglass_bottleneck_channels,use_skip = args.use_skip, depthwise_conv=args.depthwise_conv, prune_prob=args.prune_prob, pruning_method=args.pruning_method)
    else:
        net = UNetDict(in_channel=3, out_channel=args.num_landmarks, hour_glass_depth=args.Hourglass_depth, bottle_neck_channels=args.Hourglass_bottleneck_channels,use_skip = args.use_skip, depthwise_conv=args.depthwise_conv, prune_prob=args.prune_prob, pruning_method=args.pruning_method)

    if(args.use_f16):
        net.half()  # convert to half precision
        #for layer in net.modules():
        #    if isinstance(layer, nn.BatchNorm2d):
        #        layer.float()

    print("Number of weights : {}\n".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
    criterion = torch.nn.MSELoss()
    if (args.use_f16):
        #optimizer = Adam16(params=net.parameters(), lr=0.0001)
        #optimizer = torch.optim.SGD(params=net.parameters(), lr=0.01, momentum=0.9)
        optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0001, eps=1e-04)
    else:
        optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0001)

else:
    print("Data Set not supported")
    raise Exception

conv_experiment = ExperimentBuilder(network_model=net, use_gpu=args.use_gpu,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    save_model_per_n_epochs=args.save_model_per_n_epochs,
                                    rbf_width=args.rbf_width,
                                    continue_from_epoch=args.continue_from_epoch,
                                    use_tqdm=args.use_tqdm,
                                    data_provider=data_provider,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data,
                                    criterion=criterion,
                                    prune_prob=args.prune_prob,
                                    pruning_method=args.pruning_method,
                                    patience=args.patience,
                                    normalisation=args.normalisation,
                                    use_f16=args.use_f16,
                                    optimizer=optimizer
                                    )  # build an experiment object

if args.use_case == "render":


    if (test_data.inputs.shape[0] < args.no_images_to_render):
        raise ValueError(f"Size of test data set {test_data.inputs.shape[0]} to small to render {args.no_images_to_render} images")
    conv_experiment.render(data=test_data, number_images=args.no_images_to_render)
else:
    experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics


