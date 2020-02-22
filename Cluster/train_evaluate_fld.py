import numpy as np
import os
import data_providers as data_providers
from arg_extractor import get_args
from data_augmentations import Cutout
from experiment_builder import ExperimentBuilder
from model_architectures import FLDNetwork, UNet
from data_loaderUNet import UNetDataset
from data_loader300W_heat_map import Dataset300WHM


args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

import torch

torch.manual_seed(seed=args.seed)  # sets pytorch's seed

if args.dataset_name == 'Youtube' :
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

    train_data = data_providers.YoutubeDataProvider('train', batch_size=args.batch_size,
                                                    rng=rng,
                                                    max_size=max_size,
                                                    filepath_to_data=filepath_to_data)  # initialize our rngs using the argument set seed
    val_data = data_providers.YoutubeDataProvider('valid', batch_size=args.batch_size,
                                                  rng=rng,
                                                  max_size=max_size,
                                                  filepath_to_data=filepath_to_data)  # initialize our rngs using the argument set seed
    test_data = data_providers.YoutubeDataProvider('test', batch_size=args.batch_size,
                                                   rng=rng,
                                                   max_size=max_size,
                                                   filepath_to_data=filepath_to_data)  # initialize our rngs using the argument set seed

    net = FLDNetwork(  # initialize our network object, in this case a ConvNet
        input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_height),
        dim_reduction_type=args.dim_reduction_type, num_filters=args.num_filters, num_layers=args.num_layers,
        use_bias=False)
    criterion = torch.nn.MSELoss()
    optimizer= torch.optim.Adam(params=net.model.parameters(), lr=0.001)


elif args.dataset_name == 'UNet' or args.dataset_name == '300W':
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
    if args.dataset_name == 'UNet':
        dataset = UNetDataset(
                                          root_dir=os.path.join(filepath_to_data),
                                          width_in=width_in, height_in=height_in, width_out=width_out,
                                          height_out=height_out,
                                          max_size=max_size)
    elif args.dataset_name == '300W':
        dataset = Dataset300WHM(
                                      root_dir=os.path.join(filepath_to_data),
                                      width_in=width_in, height_in=height_in, width_out=width_out,
                                      height_out=height_out,
                                      num_landmarks=1,
                                      max_size=max_size)
    else:
        raise ValueError

    train_data = data_providers.BOEDataProvider(dataset=dataset, which_set='train', batch_size=args.batch_size,
                                                rng=rng)  # initialize our rngs using the argument set seed
    val_data = data_providers.BOEDataProvider(dataset=dataset, which_set='valid', batch_size=args.batch_size,
                                              rng=rng)  # initialize our rngs using the argument set seed
    test_data = data_providers.BOEDataProvider(dataset=dataset, which_set='test', batch_size=args.batch_size,
                                               rng=rng)  # initialize our rngs using the argument set seed

    net= UNet(in_channel=1, out_channel=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.99)

else:
    print("Data Set not supported")
    raise Exception

conv_experiment = ExperimentBuilder(network_model=net, use_gpu=args.use_gpu,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    continue_from_epoch=args.continue_from_epoch,
                                    use_tqdm = args.use_tqdm,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data,
                                    criterion=criterion,
                                    optimizer=optimizer
                                    )  # build an experiment object


experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
