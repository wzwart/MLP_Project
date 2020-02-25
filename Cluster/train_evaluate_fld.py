import numpy as np
import os
import data_providers as data_providers
from arg_extractor import get_args
from data_augmentations import Cutout
from experiment_builder import ExperimentBuilder
from unet import UNet
from unet_shallow import UNet_shallow
from basic_detector_net import BasicDetectorNetwork
from data_set_300WHM import Dataset300WHM
from data_set_YoutubeHM import DatasetYoutubeHM
from data_set_BOE import DatasetBOE
from data_set_youtube import DatasetYoutube

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment

import torch

torch.manual_seed(seed=args.seed)  # sets pytorch's seed

if args.dataset_name == 'Youtube':
    if os.path.isdir(args.filepath_to_data_1):
        filepath_to_data = args.filepath_to_data_1

        print(f"No1 {args.filepath_to_data_1} exists")

    elif os.path.isdir(args.filepath_to_data_2):
        filepath_to_data = args.filepath_to_data_2
        print(f"No2 {args.filepath_to_data_2} exists")

    else:
        raise FileExistsError

    try:
        max_size_dataset = int(args.max_size_data_set)
    except:
        max_size_dataset = None
        pass

    data_provider = data_providers.DataProviderYoutube
    train_data = data_provider('train', batch_size=args.batch_size,
                               rng=rng,
                               max_size=max_size_dataset,
                               filepath_to_data=filepath_to_data)  # initialize our rngs using the argument set seed
    val_data = data_providers('valid', batch_size=args.batch_size,
                              rng=rng,
                              max_size=max_size_dataset,
                              filepath_to_data=filepath_to_data)  # initialize our rngs using the argument set seed
    test_data = data_providers('test', batch_size=args.batch_size,
                               rng=rng,
                               max_size=max_size_dataset,
                               filepath_to_data=filepath_to_data)  # initialize our rngs using the argument set seed

    net = BasicDetectorNetwork(  # initialize our network object, in this case a ConvNet
        input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_height),
        dim_reduction_type=args.dim_reduction_type, num_filters=args.num_filters, num_layers=args.num_layers,
        use_bias=False)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)


elif args.dataset_name == 'BOE' or args.dataset_name == '300WHM' or args.dataset_name == 'YoutubeHM':
    if os.path.isdir(args.filepath_to_data_1):
        filepath_to_data = args.filepath_to_data_1

        print(f"No1 {args.filepath_to_data_1} exists")

    elif os.path.isdir(args.filepath_to_data_2):
        filepath_to_data = args.filepath_to_data_2
        print(f"No2 {args.filepath_to_data_2} exists")

    else:
        raise FileExistsError

    try:
        max_size_dataset = int(args.max_size_dataset)
    except:
        max_size_dataset = -1
        pass

    width_in = 128
    height_in = 128
    width_out = 128
    height_out = 128
    if args.dataset_name == 'BOE':
        dataset = DatasetBOE(
            root_dir=os.path.join(filepath_to_data),
            width_in=width_in, height_in=height_in, width_out=width_out,
            height_out=height_out,
            max_size=max_size_dataset)
    elif args.dataset_name == '300WHM':
        dataset = Dataset300WHM(
            root_dir=os.path.join(filepath_to_data),
            width_in=width_in, height_in=height_in, width_out=width_out,
            height_out=height_out,
            num_landmarks=args.num_landmarks,
            landmarks_collapsed=args.landmarks_collapsed,
            max_size=max_size_dataset)
    elif args.dataset_name == 'YoutubeHM':
        dataset = DatasetYoutubeHM(
            root_dir=os.path.join(filepath_to_data),
            width_in=width_in, height_in=height_in, width_out=width_out,
            height_out=height_out,
            num_landmarks=args.num_landmarks,
            landmarks_collapsed=args.landmarks_collapsed,
            max_size=max_size_dataset)
    else:
        raise ValueError

    data_provider = data_providers.DataProviderBOE

    train_data = data_provider(dataset=dataset, which_set='train', batch_size=args.batch_size,
                               rng=rng)  # initialize our rngs using the argument set seed
    val_data = data_provider(dataset=dataset, which_set='valid', batch_size=args.batch_size,
                             rng=rng)  # initialize our rngs using the argument set seed
    test_data = data_provider(dataset=dataset, which_set='test', batch_size=args.batch_size,
                              rng=rng)  # initialize our rngs using the argument set seed

    data_provider=train_data
    if args.dataset_name == 'BOE':
        net = UNet(in_channel=1, out_channel=2)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        if args.landmarks_collapsed:
            net = UNet_shallow(in_channel=3, out_channel=1)
        else:
            net = UNet_shallow(in_channel=3, out_channel=args.num_landmarks)
        criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.99)

else:
    print("Data Set not supported")
    raise Exception

conv_experiment = ExperimentBuilder(network_model=net, use_gpu=args.use_gpu,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    continue_from_epoch=args.continue_from_epoch,
                                    use_tqdm=args.use_tqdm,
                                    data_provider=data_provider,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data,
                                    criterion=criterion,
                                    optimizer=optimizer
                                    )  # build an experiment object

if args.use_case == "render":
    conv_experiment.render(data=train_data, number_images=args.no_images_to_render, x_y_only=False)
else:
    experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
