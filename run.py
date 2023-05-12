import os
import sys

DISABLE_GPU = False
if DISABLE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.keras.backend.clear_session()

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


import math
import numpy as np
import pandas as pd
import model_utils.models as model_utils
import model_utils.transformer as transformer
import model_utils.continual_learning as CL
from model_utils.visualization import Visualization as visualization
from model_utils.loss import LossFunction
from time import gmtime, strftime
import visualkeras
import h5py
from matplotlib import pyplot as plt
from torchviz import make_dot
import time
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as K

from data_utils.get_input import Data_IO
from data_utils.preprocessing import PreProcesser
from config import global_params, prediction, model_configs

data_path = global_params["data_path"]

os.environ['TZ'] = "Asia/Singapore"
time.tzset()

import torch
import torch.nn as nn
import math
import copy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--delay', help='number of time steps', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

def train_model_tf(model_name, cfg_name, data_io, plot=False, new_window_size=None):
    # data_io = Data_IO(data_path=data_path)
    prediction_cfg = prediction[cfg_name]
    select_dim = global_params["output_dim"]
    print(__file__, sys._getframe().f_lineno, "\n---------->", data_io.dataframe.columns[select_dim])

    window_size = new_window_size if new_window_size else prediction_cfg["window_size"]
    num_of_var = len(select_dim)
    x_train, y_train, x_test, y_test = data_io.io_window_xy(select_dim=select_dim, 
        window_size=window_size, num_of_step=1, delay=args.delay, split_ratio=0.7, shuffle=False)
    x_train, y_train, x_test, y_test = \
        np.nan_to_num(x_train), np.nan_to_num(y_train), np.nan_to_num(x_test), np.nan_to_num(y_test)

    y_train = y_train.reshape((y_train.shape[0], y_train.shape[2]))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[2]))

    print(__file__, sys._getframe().f_lineno, "\n---------->",
          x_train.shape, y_train.shape, "\n", x_test.shape, y_test.shape)

    model_obj = model_utils.Models(model_name=model_name)
    model_params = [(x_train.shape[1], x_train.shape[2]), len(select_dim)]
    if cfg_name == "efficientnetv2_rnn": model_params += [prediction_cfg]
    model = model_obj.load_model(model_params)

    # Confirm unfrozen layers
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)

    os.mkdir("./models") if not os.path.exists("./models") else print("./models already exists")
    os.mkdir("./logs") if not os.path.exists("./logs") else print("./logs already exists")

    plot_model(model, to_file="./models/" + model_name + time.strftime("_%Y-%m-%d_%H-%M-%S") + ".png", 
               show_shapes=True, show_layer_names=True)
    # visualkeras.layered_view(model, legend=True)

    model_path = "./models/" + model_name + time.strftime("_%Y-%m-%d_%H-%M-%S") + ".h5"

    tb_callback = TensorBoard(log_dir='./logs/tensorboard_' + model_name + 
                                  time.strftime("_%Y-%m-%d_%H-%M-%S"), # log directory
                              histogram_freq=0,  # Calculate the histogram according to epoch, 0 is not calculated
                              write_graph=True,  # Whether to store the network structure diagram
                              write_images=True, # Whether to visualize parameters
                              embeddings_freq=0, 
                              embeddings_layer_names=None, 
                              embeddings_metadata=None)
    train_callbacks = [tb_callback, 
                       EarlyStopping(monitor='val_loss', patience=50, verbose=1), 
                       ModelCheckpoint(model_path, save_best_only=True)]

    # custom function example
    model.compile(optimizer="Nadam", loss=LossFunction.rmse, metrics=[LossFunction.r_square])

    epochs = prediction_cfg["epochs"]
    learning_rate = prediction_cfg["learning_rate"]
    batch_size = prediction_cfg["epochs"]
    decay_rate = prediction_cfg["decay_rate"]
    K.set_value(model.optimizer.learning_rate, learning_rate)
    K.set_value(model.optimizer.decay, decay_rate)

    history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(x_test, y_test), 
                        callbacks=train_callbacks, 
                        verbose=1, shuffle=True)
    
    if plot: visualization.tr_plot(history, 0)

    model = tf.keras.models.load_model(model_path, 
                                       custom_objects={"rmse": LossFunction.rmse, 
                                                       "r_square": LossFunction.r_square,
                                                       "mse": LossFunction.mse})

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(__file__, sys._getframe().f_lineno, "---------->\n",
          'Loss    :', test_loss)
    print('Accuracy:', test_acc) 

    y_pred = model.predict(x_test)

    if plot: visualization.plot_pred_true(y_pred, y_test, select_dim)

    return history, [test_loss, test_acc]

def train_model_trans(model_name, cfg_name, data_io, plot=False, train=True):
    select_dim = global_params["select_dim"]
    window_size = global_params["window_size"]
    result = []

    x_train, y_train, x_test, y_test = data_io.io_sequence_xy(select_dim=select_dim, \
        window_size=window_size, num_of_step=1, delay=args.delay, split_ratio=0.7, shuffle=True)
    x_train, y_train, x_test, y_test = \
        np.nan_to_num(x_train), np.nan_to_num(y_train), np.nan_to_num(x_test), np.nan_to_num(y_test)
    print(__file__, sys._getframe().f_lineno, "\n---------->", 
          x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_data, test_data = PreProcesser.transfer_to_sequence(x_train, y_train, device), \
        PreProcesser.transfer_to_sequence(x_test, y_test, device)
    print(__file__, sys._getframe().f_lineno, "\n---------->",
          train_data.shape, test_data.shape)
    model_obj = model_utils.Models(model_name=model_name)
    
    model = model_obj.load_model([prediction[cfg_name]["feature_size"], 
                                  prediction[cfg_name]["num_of_var"], 
                                  prediction[cfg_name]["select_dim"],
                                  prediction[cfg_name]["num_layers"],
                                  prediction[cfg_name]["num_head"],
                                  prediction[cfg_name]["dropout"]])
    print(model)

    batch_size = global_params["batch_size"]
    loss_criterion = LossFunction.rmse_torch
    accuracy_criterion = LossFunction.r_square_torch
    learning_rate = prediction[cfg_name]["learning_rate"]
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - prediction[cfg_name]["decay_rate"]))
    best_val_loss = float("inf")
    epochs = prediction[cfg_name]["epochs"]
    best_model = None
    model_save = os.path.join("./models", model_name + ".pth")

    history = {
        'r_square': [], 'loss': [], 'val_r_square': [], 'val_loss': []
    }

    if train == False:
        model.load_state_dict(torch.load(model_save))
        best_model = copy.deepcopy(model)
        test_loss, test_acc, y_pred = transformer.TransUtils.validate(best_model, test_data, batch_size, loss_criterion, accuracy_criterion)
        result.append([test_loss, test_acc])
        if plot: 
            visualization.plot_pred_true(y_pred, y_test[:, :, -1], select_dim)
            visualization.plot_pred_true_partraw(y_pred, y_test[:, :, -1], select_dim)
        return None, result[0]

    training_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss, train_acc = transformer.TransUtils.train(model, train_data, batch_size, optimizer, loss_criterion, accuracy_criterion) 
        test_loss, test_acc, _ = transformer.TransUtils.validate(model, test_data, batch_size, loss_criterion, accuracy_criterion)
        history['r_square'].append(train_acc)
        history['loss'].append(train_loss)
        history['val_r_square'].append(test_acc)
        history['val_loss'].append(test_loss)
    
        print('-' * 107)
        print('| end of epoch {:3d} | time: {:5.2f}s | train acc & loss {:5.5f}, {:5.5f} | valid acc & loss {:5.5f}, {:5.5f} |'.format(epoch, (time.time() - epoch_start_time),
                                        train_acc, train_loss, test_acc, test_loss))

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_model = copy.deepcopy(model)

        scheduler.step()

    print('-' * 100, '\n Training process done in: {:5.2f}s.'.format(time.time() - training_start_time))

    test_loss, test_acc, y_pred = transformer.TransUtils.validate(best_model, test_data, batch_size, loss_criterion, accuracy_criterion)
    result.append([test_loss, test_acc])
    torch.save(best_model.state_dict(), model_save)
    print(__file__, sys._getframe().f_lineno, "---------->\n",
          "model's state_dict is saved in:", model_save, 
          "\nresults of bset model:", test_loss, test_acc)
    if plot: 
        visualization.tr_plot(history, 0)
        visualization.plot_pred_true(y_pred, y_test[:, :, -1], select_dim)
        visualization.plot_pred_true_partraw(y_pred, y_test[:, :, -1], select_dim)

    return None, result[0]

def train_model_cl(model_name, cfg_name, data_io, plot=False, train=True, softmode=0):
    prediction_cfg = prediction[cfg_name]
    select_dim = global_params["select_dim"]
    window_size = prediction_cfg["seq_length"]
    result = []

    if softmode == 0:
        x_train, y_train, x_test, y_test = data_io.io_window_xy_CL(select_dim=select_dim, 
            window_size=window_size, num_of_step=1, delay=args.delay, split_ratio=0.7, shuffle=False)
    elif softmode == 1:
        x_train, y_train, x_test, y_test = data_io.io_window_xy_CL_soft(select_dim=select_dim, 
            window_size=window_size, num_of_step=1, delay=args.delay, split_ratio=0.7, shuffle=False)
    else:
        x_train, y_train, x_test, y_test = data_io.io_window_xy_CL_all(select_dim=select_dim, 
            window_size=window_size, num_of_step=1, delay=args.delay, split_ratio=0.7, shuffle=False)
    x_train, y_train, x_test, y_test = \
        np.nan_to_num(x_train), np.nan_to_num(y_train), np.nan_to_num(x_test), np.nan_to_num(y_test)

    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[2], -1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[2], -1))

    print(__file__, sys._getframe().f_lineno, "\n---------->",x_train.shape, y_train.shape, "\n", x_test.shape, y_test.shape)

    model_obj = model_utils.Models(model_name=model_name)
    model = model_obj.load_model([prediction_cfg["seq_length"], 
                                    prediction_cfg["seq_length_long"],
                                    prediction_cfg["output_dim_1"],
                                    prediction_cfg["output_dim_2"],
                                    prediction_cfg["output_dim"],
                                    prediction_cfg["input_dim_new"],
                                    prediction_cfg["feature_size"],
                                    prediction_cfg["num_heads"],
                                    prediction_cfg["dropout"],
                                    prediction_cfg["num_layers"]]).to(device)
    print(model)

    # optimizer, loss function, memory manager, and training & testing process
    batch_size = prediction_cfg["batch_size"]
    loss_criterion = LossFunction.rmse_torch_CL
    accuracy_criterion = LossFunction.r_square_torch_CL
    learning_rate = prediction_cfg["learning_rate"]
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - prediction_cfg["decay_rate"]))
    best_val_loss = float("inf")
    epochs = prediction_cfg["epochs"]
    best_model = None
    model_save = os.path.join("./models", model_name + ".pth")

    history = {
        'r_square': [], 'loss': [], 'val_r_square': [], 'val_loss': []
    }

    if train == False:
        model.load_state_dict(torch.load(model_save))
        best_model = copy.deepcopy(model)
        memory_test = CL.memory(max_length=prediction_cfg["seq_length_long"], channel=prediction_cfg["input_dim_new"])
        test_loss, test_acc, y_pred = CL.cl_utils.validate(best_model, (x_test, y_test), memory_test, batch_size, loss_criterion, accuracy_criterion)
        result.append([test_loss, test_acc])
        if plot: 
            visualization.plot_pred_true(y_pred, y_test, select_dim)
        return None, result[0]

    training_start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        memory_train = CL.memory(max_length=prediction_cfg["seq_length_long"], channel=prediction_cfg["input_dim_new"])
        memory_test = CL.memory(max_length=prediction_cfg["seq_length_long"], channel=prediction_cfg["input_dim_new"])
        train_loss, train_acc = CL.cl_utils.train(model, (x_train, y_train), memory_train, batch_size, optimizer, loss_criterion, accuracy_criterion) 
        test_loss, test_acc, _ = CL.cl_utils.validate(model, (x_test, y_test), memory_test, batch_size, loss_criterion, accuracy_criterion)
        history['r_square'].append(train_acc)
        history['loss'].append(train_loss)
        history['val_r_square'].append(test_acc)
        history['val_loss'].append(test_loss)
    
        print('-' * 107)
        print('| end of epoch {:3d} | time: {:5.2f}s | train acc & loss {:5.5f}, {:5.5f} | valid acc & loss {:5.5f}, {:5.5f} |'.format(epoch, (time.time() - epoch_start_time),
                                        train_acc, train_loss, test_acc, test_loss))

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_model = copy.deepcopy(model)

        scheduler.step()

    print('-' * 100, '\n Training process done in: {:5.2f}s.'.format(time.time() - training_start_time))

    new_memory = CL.memory(max_length=prediction_cfg["seq_length_long"], channel=prediction_cfg["input_dim_new"])
    test_loss, test_acc, y_pred = CL.cl_utils.validate(best_model, (x_test, y_test), new_memory, batch_size, loss_criterion, accuracy_criterion)
    result.append([test_loss, test_acc])
    torch.save(best_model.state_dict(), model_save)
    print(__file__, sys._getframe().f_lineno, "---------->\n",
          "model's state_dict is saved in:", model_save, 
          "\nresults of bset model:", test_loss, test_acc)
    
    if plot: 
        visualization.tr_plot(history, 0)
        visualization.plot_pred_true(y_pred, y_test, select_dim)

    return None, result[0]


if __name__ == "__main__":
    data_io = Data_IO(data_path=data_path)
    result = {}
    
    for model_cfg in model_configs:
        print(__file__, sys._getframe().f_lineno, "\n---------->", model_cfg)
        if model_cfg[1] == "transformer":
            _, result[model_cfg[1] + "_&_" + model_cfg[0]] = train_model_trans(model_cfg[0], model_cfg[1], data_io, 
                                                                               plot=True, train=True)
        elif model_cfg[1] == "continual_learning":
            _, result[model_cfg[1] + "_&_" + model_cfg[0]] = train_model_cl(model_cfg[0], model_cfg[1], data_io, 
                                                                            plot=True, train=True, softmode=0)
        else:
            _, result[model_cfg[1] + "_&_" + model_cfg[0]] = train_model_tf(model_cfg[0], model_cfg[1], 
                                                                            data_io, plot=True)
    
    result_log_file = "./logs/results" + time.strftime("_%Y-%m-%d_%H-%M-%S") + ".txt"
    print(__file__, sys._getframe().f_lineno, "---------->\n", result, 
          "\nresults are saved in:", result_log_file)
    with open(result_log_file, "w") as f:
        f.write("{\n")
        for model in result:
            f.write("\t\'" + model + "\': " + str(result[model]) + ",\n")
        f.write("}\n")
