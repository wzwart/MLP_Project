import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time
from matplotlib import cm

from torch.optim.adam import Adam

from storage_utils import save_statistics

class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, data_provider,train_data, val_data,
                 test_data, use_gpu, criterion, optimizer, use_tqdm=True, continue_from_epoch=-1):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = experiment_name
        self.model = network_model
        self.model.reset_parameters()
        try:
            self.device = torch.cuda.current_device()
        except:
            self.device = torch.device('cpu')
            pass

        self.use_tqdm =use_tqdm
        self.criterion=criterion.to(self.device)
        self.continue_from_epoch=continue_from_epoch
        self.data_provider=data_provider
        try:
            self.device = torch.cuda.current_device()
            print("number of available devicews", torch.cuda.device_count())
            print("Use GPU", use_gpu)
        except:
            self.device = torch.device('cpu')
            print("No CUDA installed, use CPU")
            use_gpu=False
            pass
        try:
            if torch.cuda.device_count() > 1 and use_gpu:
                self.device = torch.cuda.current_device()
                self.model.to(self.device)
                self.model = nn.DataParallel(module=self.model)
                print('Use Multi GPU', self.device)
            elif torch.cuda.device_count() == 1 and use_gpu:
                self.device = torch.cuda.current_device()
                self.model.to(self.device)  # sends the model from the cpu to the gpu
                print('Use GPU', self.device)
            else:
                print("use CPU")
                self.device = torch.device('cpu')  # sets the device to be CPU
                print(self.device)
        except:
            print("CUDA not installed")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)
            pass

        # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data


        self.optimizer =optimizer

        print('System learnable parameters')
        num_conv_layers = 0
        num_linear_layers = 0
        total_num_parameters = 0
        for name, value in self.named_parameters():
            print(name, value.shape)
            if all(item in name for item in ['conv', 'weight']):
                num_conv_layers += 1
            if all(item in name for item in ['linear', 'weight']):
                num_linear_layers += 1
            total_num_parameters += np.prod(value.shape)



        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs)
        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_loss = 1000000

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = num_epochs
        if self.continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_loss, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best val model index
                # and the best val acc of that model
                self.starting_epoch = self.state['current_epoch_idx']
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()

        elif self.continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_loss, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=self.continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']

        else:
            self.starting_epoch = 0
            self.state = dict()

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def generateHeatmap(self, center_x, center_y, width, height):
        x = np.arange(width)
        y = np.arange(height)
        xv, yv = np.meshgrid(x, y)
        width_norm = 0.07 * np.sqrt(width * height)
        hm = np.exp(-0.5 * ((xv - center_x) ** 2 + (yv - center_y) ** 2) / (width_norm ** 2))
        # hm = hm - hm.mean(axis=(0, 1))
        # but don't normalize variance
        return hm

    def NME(self, true, predicted):

        bbox = np.max(true, axis=0) - np.min(true, axis=0)
        d = np.sqrt(bbox[0]*bbox[1])
        return np.sum(np.square(true-predicted))/d

    def compute_nme(self, out, p):

        set1 = cm.get_cmap('Set1')
        colors = np.asarray(set1.colors)
        no_colors = colors.shape[0]
        no_landmarks = out.shape[3]

        nme = 0
        count = 0
        height_out = out.shape[1]
        width_out = out.shape[2]
        n_images = out.shape[0]
        u = np.zeros((width_out, height_out, 3))
        u[:, :, 0] = self.generateHeatmap(int(width_out / 2), int(height_out / 2), width_out, height_out)
        u[:, :, 1] = self.generateHeatmap(int(width_out / 2), int(height_out / 2), width_out, height_out)
        u[:, :, 2] = self.generateHeatmap(int(width_out / 2), int(height_out / 2), width_out, height_out)

        u = np.clip(u, 0, 1)

        u = np.array([u.transpose((2, 0, 1))])
        u = torch.Tensor(u).float().to(self.device)
        for row_num in range(n_images):
            if type(out) != type(None):
                predicted = np.array([])
                p_img = p[row_num]
                for i in range(no_landmarks):
                    out_img = (out[row_num] - out[row_num].min())
                    #print(out_img.shape)
                    t = torch.Tensor(3, out_img.shape[0], out_img.shape[1])
                    t[0] = out_img[:, :, i] * colors[i % no_colors, 0]
                    t[1] = out_img[:, :, i] * colors[i % no_colors, 1]
                    t[2] = out_img[:, :, i] * colors[i % no_colors, 2]
                    t = t.permute((1, 2, 0))

                    out_conv = t.permute((2, 0, 1)).unsqueeze(0)
                    out_conv = out_conv.to(self.device)
                    padded_tensor = F.pad(out_conv, (
                        int(height_out / 2), int(height_out / 2) - 1, int(width_out / 2),
                        int(width_out / 2) - 1))
                    cross_corr = F.conv2d(padded_tensor, u, padding=0)

                    argmax = torch.argmax(cross_corr)

                    index_i, index_j = np.unravel_index(argmax.detach().cpu(),
                                                        (out_img.shape[0],out_img.shape[1]))
                    if (i == 0):
                        predicted = np.hstack((predicted, np.array([index_j, index_i])))
                    else:
                        predicted = np.vstack((predicted, np.array([index_j, index_i])))

                nme += self.NME(p_img, predicted)
                count += 1

        return nme/count

    def run_train_iter(self, x, y, p):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """

        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        # if len(y.shape) > 1:
        #     y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels

        #print(type(x))

        if type(x) is np.ndarray:
            x= torch.Tensor(x).float().to(device=self.device)
            y =torch.Tensor(y).float().to(device=self.device)  # send data to device as torch tensors

        x = x.to(self.device)
        y = y.to(self.device)
        out = self.model.forward(x)  # forward the data in the model

        if str(self.criterion)=="CrossEntropyLoss()":
            loss_in = out.reshape((out.shape[0] * out.shape[1] * out.shape[2], out.shape[3]))
            loss_target = y.reshape((y.shape[0] * y.shape[1] * y.shape[2], y.shape[3]))[:, 0]
        else:
            loss_in=out
            loss_target=y

        loss = self.criterion(loss_in, loss_target)

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        # _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        #
        # accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        #nme = self.compute_nme(out, p)

        return loss.data.detach().cpu().numpy(), 3

    def run_evaluation_iter(self, x, y, p):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode
        # if len(y.shape) > 1:
        #     y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
        if type(x) is np.ndarray:
            x = torch.Tensor(x).float().to(device=self.device)
            y = torch.Tensor(y).float().to(device=self.device)  # convert data to pytorch tensors and send to the computation device

        x = x.to(self.device)
        y = y.to(self.device)
        out = self.model.forward(x)  # forward the data in the model

        if str(self.criterion)=="CrossEntropyLoss()":
            loss_in = out.reshape((out.shape[0] * out.shape[1] * out.shape[2], out.shape[3]))
            loss_target = y.reshape((y.shape[0] * y.shape[1] * y.shape[2], y.shape[3]))[:, 0]
        else:
            loss_in=out
            loss_target=y

        loss = self.criterion(loss_in, loss_target)

        #nme = self.compute_nme(out, p)

        return loss.data.detach().cpu().numpy(), 3

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to s The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        state['network'] = self.state_dict()  # save network parameter and other variables.
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def run_training_epoch(self, epoch_idx, current_epoch_losses):
        if not self.use_tqdm :
            f = open("log.log", "w")
        else:
            f = sys.stdout
        with tqdm.tqdm(total=len(self.train_data), file=f) as pbar_train:  # create a progress bar for training
            running_loss=0
            for idx, (x,y,p)  in enumerate(self.train_data):  # get data batches
                loss, accuracy = self.run_train_iter(x=x, y=y, p=p)  # take a training iter step
                current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                current_epoch_losses["train_nme"].append(accuracy)  # add current iter acc to the train acc list
                if self.use_tqdm:
                    pbar_train.update(1)
                    pbar_train.set_description("Epoch {}: Train     loss: {}".format(epoch_idx,loss))
                else:
                    running_loss += loss.item()
                    if idx % 10 == 9 and not self.use_tqdm:  # print every 10 batches
                        print('Train epoch {}, Batch: {}, Avg. Loss: {}'.format(epoch_idx, idx + 1, running_loss / 10))
                        running_loss = 0.0
        return current_epoch_losses

    def run_validation_epoch(self, epoch_idx, current_epoch_losses):
        if not self.use_tqdm :
            f = open("log.log", "w")
        else:
            f = sys.stdout
        running_loss = 0
        with tqdm.tqdm(total=len(self.val_data), file=f) as pbar_val:  # create a progress bar for validation
            for idx, (x, y, p) in enumerate(self.val_data):  # get data batches

                loss, accuracy = self.run_evaluation_iter(x=x, y=y, p=p)  # run a validation iter
                current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                current_epoch_losses["val_nme"].append(accuracy)  # add current iter acc to val acc lst.
                if self.use_tqdm:
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description(
                        "Epoch {}: Validate  loss: {}".format(epoch_idx, loss))
                else:
                    running_loss += loss.item()
                    if idx % 10 == 9 and not self.use_tqdm:  # print every 10 batches
                        print('Validate Epoch {}, Batch: {}, Avg. Loss: {}'.format(epoch_idx, idx + 1, running_loss / 10))
                        running_loss = 0.0

        return current_epoch_losses

    def run_testing_epoch(self, current_epoch_losses):
        running_loss = 0
        if not self.use_tqdm :
            f = open("log.log", "w")
        else:
            f = sys.stdout
        running_loss = 0

        with tqdm.tqdm(total=len(self.test_data), file=f) as pbar_test:  # ini a progress bar
            for idx, (x, y, p) in enumerate(self.test_data):  # sample batch

                loss, accuracy = self.run_evaluation_iter(x=x,
                                                          y=y, p=p)  # compute loss and accuracy by running an evaluation step
                current_epoch_losses["test_loss"].append(loss)  # save test loss
                current_epoch_losses["test_nme"].append(accuracy)  # save test accuracy
                if self.use_tqdm:
                    pbar_test.update(1)  # update progress bar status
                    pbar_test.set_description(
                        "Test: loss: {}".format(loss))  # update progress bar string output
                else:
                    running_loss += loss.item()
                    if idx % 10 == 9 and not self.use_tqdm:  # print every 10 batches
                        print('Test Batch: {}, Avg. Loss: {}'.format(idx + 1, running_loss / 10))
                        running_loss = 0.0
        return current_epoch_losses

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_acc'], state

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_nme": [], "train_loss": [], "val_nme": [],
                        "val_loss": [], "curr_epoch": []}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):

            epoch_start_time = time.time()
            current_epoch_losses = {"train_nme": [], "train_loss": [], "val_nme": [], "val_loss": []}
            current_epoch_losses = self.run_training_epoch(epoch_idx,current_epoch_losses)
            current_epoch_losses = self.run_validation_epoch(epoch_idx, current_epoch_losses)
            val_mean_loss = np.mean(current_epoch_losses['val_loss'])
            if val_mean_loss < self.best_val_model_loss:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_loss = val_mean_loss  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx
            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(value))
                # get mean of all metrics of current epoch metrics dict,

                # to get them ready for storage and output on the terminal.

            total_losses['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False) # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_acc'] = self.best_val_model_loss
            self.state['best_val_model_idx'] = self.best_val_model_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest', state=self.state)

        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        # load best validation model
                        model_save_name="train_model")
        current_epoch_losses = {"test_nme": [], "test_loss": []}  # initialize a statistics dict
        print(self.best_val_model_idx)
        current_epoch_losses = self.run_testing_epoch(current_epoch_losses=current_epoch_losses)

        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format

        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        return total_losses, test_losses

    def render(self,data,number_images, x_y_only=False ):
        '''
        Loads the model and then class the render function of the data set

        Args:
            epoch: the epoch from which we load, -2 for the last one

        Returns:

        '''
        if not x_y_only:
            if self.continue_from_epoch == -2:
                try:
                    self.best_val_model_idx, self.best_val_model_loss, self.state = self.load_model(
                        model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                        model_idx='latest')  # reload existing model from epoch and return best val model index
                    # and the best val acc of that model
                    self.starting_epoch = self.state['current_epoch_idx']
                except:
                    print("Model objects cannot be found, initializing a new model and starting from scratch")
                    self.starting_epoch = 0
                    self.state = dict()

            elif self.continue_from_epoch != -1:  # if continue from epoch is not -1 then
                self.best_val_model_idx, self.best_val_model_loss, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx=self.continue_from_epoch)  # reload existing model from epoch and return best val model index
                # and the best val acc of that model
                self.starting_epoch = self.state['current_epoch_idx']
            else:
                raise ValueError(f"Can not load from epoch {self.continue_from_epoch}")

            self.model.eval()
        for (x,y,p) in data: #is only executed once, data can only be accessed through an enumerator
            if type(x) is np.ndarray:

                x_net =  torch.Tensor(x).float()[:number_images].to(device=self.device)
                x_img = x[:number_images].copy()
                y_img = y[:number_images].copy()
                p_img = p[:number_images].copy()
            else:
                x_net=x.copy()
                x_img=x.detach().cpu().numpy()
                y_img=y.detach().cpu().numpy()
                p_img=p.detach().cpu().numpy()
            if  x_y_only:
                out=None
            else:
                out = self.model.forward(x_net) # forward the data in the model

                if str(self.criterion) == "CrossEntropyLoss()":
                    loss_in = out.reshape((out.shape[0] * out.shape[1] * out.shape[2], out.shape[3]))
                    loss_target = y.reshape((y.shape[0] * y.shape[1] * y.shape[2], y.shape[3]))[:, 0]
                else:
                    loss_in = out
                    loss_target = y

                loss = self.criterion(torch.Tensor(loss_in).float().to(device=self.device), torch.Tensor(loss_target).float().to(device=self.device))
                print(loss)
                out = out.detach().cpu().numpy()  # forward the data in the model



            break
        self.data_provider.render(x=x_img,y=y_img,p=p_img,out=out,number_images=number_images)

