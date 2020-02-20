import numpy as np
import os
import data_providers as data_providers
from arg_extractor import get_args
from data_augmentations import Cutout
from experiment_builder import ExperimentBuilder
from model_architectures import FLDNetwork

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

import torch

torch.manual_seed(seed=args.seed)  # sets pytorch's seed


if args.dataset_name == '300W':
    if os.path.isdir(args.filepath_to_data_1):
        filepath_to_data=args.filepath_to_data_1

        print(f"No1 {args.filepath_to_data_1} exists")

    elif os.path.isdir(args.filepath_to_data_2):
        filepath_to_data=args.filepath_to_data_2
        print(f"No2 {args.filepath_to_data_2} exists")

    else:
        raise FileExistsError

    try:
        max_size = int(args.max_size)
    except:
        max_size=None
        pass

    train_data = data_providers.FacesDataProvider('train', batch_size=args.batch_size,
                                                   rng=rng,
                                                  max_size=max_size,
                                                  filepath_to_data=filepath_to_data)  # initialize our rngs using the argument set seed
    val_data = data_providers.FacesDataProvider('valid', batch_size=args.batch_size,
                                                 rng=rng,
                                                max_size=max_size,
                                                filepath_to_data=filepath_to_data)  # initialize our rngs using the argument set seed
    test_data = data_providers.FacesDataProvider('test', batch_size=args.batch_size,
                                                  rng=rng,
                                                 max_size=max_size,
                                                 filepath_to_data=filepath_to_data)  # initialize our rngs using the argument set seed


elif args.dataset_name == 'UNet':
    if os.path.isdir(args.filepath_to_data_1):
        filepath_to_data=args.filepath_to_data_1

        print(f"No1 {args.filepath_to_data_1} exists")

    elif os.path.isdir(args.filepath_to_data_2):
        filepath_to_data=args.filepath_to_data_2
        print(f"No2 {args.filepath_to_data_2} exists")

    else:
        raise FileExistsError

    try:
        max_size = int(args.max_size)
    except:
        max_size=None
        pass

    width_in = 284
    height_in = 284
    width_out = 196
    height_out = 196

    train_data = data_providers.UNetDataProvider(which_set='train', batch_size=args.batch_size,
                                                   rng=rng,
                                                 width_in=width_in, height_in=height_in, width_out=width_out,
                                                 height_out=height_out,
                                                  max_size=max_size,
                                                  filepath_to_data=filepath_to_data)  # initialize our rngs using the argument set seed
    val_data = data_providers.UNetDataProvider(which_set='valid', batch_size=args.batch_size,
                                                rng=rng,
                                                width_in=width_in, height_in=height_in, width_out=width_out,
                                                height_out=height_out,
                                                max_size=max_size,
                                                filepath_to_data=filepath_to_data)  # initialize our rngs using the argument set seed
    test_data = data_providers.UNetDataProvider(which_set='test', batch_size=args.batch_size,
                                                  rng=rng,
                                                 width_in=width_in, height_in=height_in, width_out=width_out,
                                                 height_out=height_out,
                                                 max_size=max_size,
                                                 filepath_to_data=filepath_to_data)  # initialize our rngs using the argument set seed

else:
    print("Data Set not supported")
    raise Exception

custom_conv_net = FLDNetwork(  # initialize our network object, in this case a ConvNet
    input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_height),
    dim_reduction_type=args.dim_reduction_type, num_filters=args.num_filters, num_layers=args.num_layers,
    use_bias=False)

conv_experiment = ExperimentBuilder(network_model=custom_conv_net, use_gpu=args.use_gpu,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    continue_from_epoch=args.continue_from_epoch,
                                    use_tqdm = args.use_tqdm,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data)  # build an experiment object

experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
