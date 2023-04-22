import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten
import sys
import model_utils.nn_utils as nn_utils
from model_utils.transformer import TransAm
from model_utils.loss import LossFunction
from model_utils.continual_learning import continual_learning
import config
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import numpy as np
import time
import math

class Models():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model_dist = {
            "simple_mlp": self.mlp_4, # soft
            "simple_lstm": self.lstm_1_deeper_tf, # pred 1
            "cnn_rnn": self.cnn_lstm,
            "cnn_rnn_no_batchnorm": self.cnn_lstm_no_batchnorm,
            "cnn_rnn_layernorm": self.cnn_lstm_layernorm,
            "resnet_rnn": self.resnet_lstm, # pred 3 
            "resnet_rnn_no_batchnorm_2": self.resnet_lstm_no_batchnorm_2, # pred 3 
            "resnet_rnn_layernorm": self.resnet_lstm_layernorm, # pred 3 
            "efficientnetv2_rnn": self.efficientnet_lstm, # pred 4
            "transformer_1": self.transformer_1, # pred 5
            "continual_learning_1": self.continual_learning,
        }
        if model_name not in self.model_dist:
            print(__file__, sys._getframe().f_lineno, "\n---------->", "Invalid model_name :", self.model_name)
            raise KeyError

    def load_model(self, params):
        return self.model_dist[self.model_name](*params)

    @staticmethod
    def continual_learning(seq_length=500, seq_length_long=500*50, output_dim_1=256, 
                           output_dim_2=64, output_dim=1, input_dim_new=6, feature_size=512, 
                           num_heads=16, dropout=0.1, num_layers=2):
        return continual_learning(seq_length=seq_length, seq_length_long=seq_length_long, 
                                  output_dim_1=output_dim_1, output_dim_2=output_dim_2, 
                                  input_dim_new=input_dim_new,
                                  output_dim=output_dim, feature_size=feature_size, 
                                  num_heads=num_heads, dropout=dropout, num_layers=num_layers)

    @staticmethod
    def transformer_1(feature_size=250, num_of_var=6, select_dim=1, 
                      num_layers=1, nhead=10, dropout=0.1):
        return TransAm(feature_size=feature_size, num_of_var=num_of_var, 
                       select_dim=select_dim, num_layers=num_layers, 
                       nhead=nhead, dropout=dropout).to(Models.device)

    @staticmethod
    def efficientnet_lstm(input_shape, output_dim, config):
        model = Sequential()
        model.add(tf.keras.layers.Reshape(target_shape=(input_shape[0], input_shape[1], 1), 
                                         input_shape=input_shape))
        model.add(nn_utils.Conv1Dim(
            input_shape=input_shape,
            filters=32,
            kernel_size=4,
            strides=1
        ))
        model.add(Activation(tf.nn.relu))
        # model.add(nn_utils.MaxPool1Dim(
        #     pool_size=2,
        #     strides=2,
        # ))
        model.add(Dropout(
            rate=0.2,
            noise_shape=None, 
            seed=None
        ))
        model.add(tf.keras.layers.Reshape(target_shape=(input_shape[0], -1, 3), 
                                          input_shape=input_shape))
        model.add(tf.keras.applications.EfficientNetV2S(
            include_top=config["include_top"], input_shape=model.output_shape[1:], 
            weights=config["weights"], pooling=config["pooling"]))
        # model.add(Dense(units=256*input_shape[1]))
        model.add(Dense(units=256))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        # model.add(tf.keras.layers.Reshape(target_shape=(-1, input_shape[1])))
        # model.add(LSTM(units=128, input_shape=input_shape, return_sequences=False))
        # model.add(Activation("relu"))
        model.add(Dense(units=64))
        model.add(Activation("relu"))
        model.add(Dropout(0.05))
        model.add(Dense(units=output_dim))

        model.summary()
        return model

    @staticmethod
    def mlp_4(input_shape, output_dim):
        model = Sequential()
        model.add(Dense(units=1024, input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(Dense(units=1024))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(Dense(units=512))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(Dense(units=256))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(Dense(units=256))
        model.add(Activation("relu"))
        model.add(Dropout(0.05))
        model.add(Dense(units=128))
        model.add(Activation("relu"))
        model.add(Dense(units=64))
        model.add(Activation("relu"))
        model.add(Dense(units=output_dim))
        model.summary()
        return model

    @staticmethod
    def lstm_1_deeper_tf(input_shape, output_dim):
        model = Sequential()
        model.add(LSTM(units=256, input_shape=input_shape, return_sequences=False))
        model.add(Activation("relu"))
        model.add(Dense(units=128))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(Dense(units=128))
        model.add(Activation("relu"))
        model.add(Dropout(0.075))
        model.add(Dense(units=64))
        model.add(Activation("relu"))
        model.add(Dropout(0.05))
        model.add(Dense(units=output_dim))
        model.summary()
        return model

    # for prediction 2
    @staticmethod
    def cnn_lstm(input_shape, num_of_conv=1, append_rnn=True, output_dim=1):
        model = Sequential()

        # input block
        model.add(nn_utils.Conv1Dim(
            input_shape=input_shape,
            filters=32,
            kernel_size=4,
            strides=1
        ))
        model.add(nn_utils.BatchNorm(
            trainable=True
        ))
        model.add(Activation(tf.nn.relu))
        model.add(nn_utils.MaxPool1Dim(
            pool_size=2,
            strides=2,
        ))
        model.add(Dropout(
            rate=0.1,
            noise_shape=None, 
            seed=None
        ))

        # conv block
        for i in range(num_of_conv):
            model.add(nn_utils.Conv1Dim(
                input_shape=None,
                filters=64+i*32,
                kernel_size=3-i,
                strides=1
            ))
            model.add(nn_utils.BatchNorm(
                trainable=True
            ))
            model.add(Activation(tf.nn.relu))

        # lstm block
        if append_rnn:
            model.add(Dropout(
                rate=0.1,
                noise_shape=None, 
                seed=None
            ))
            model.add(nn_utils.RnnLSTM(
                units=256,
                bias=None,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=False, 
                return_state=False
            ))
        else:
            model.add(Flatten(
                data_format=None
            ))
            model.add(nn_utils.FullConn(
                units=128,
                activation=None,
                bias=None,
            ))
            model.add(Activation(tf.nn.relu))
            model.add(Dropout(
                rate=0.2,
                noise_shape=None, 
                seed=None
            ))

        # output block
        model.add(nn_utils.FullConn(
            units=64,
            activation=None,
            bias=None,
        ))
        model.add(Activation(tf.nn.relu))
        model.add(Dropout(
            rate=0.1,
            noise_shape=None, 
            seed=None
        ))
        model.add(nn_utils.FullConn(
            units=16,
            activation=None,
            bias=None,
        ))
        model.add(Activation(tf.nn.relu))
        # model.add(Dropout(
        #     rate=0.1,
        #     noise_shape=None, 
        #     seed=None
        # ))
        model.add(nn_utils.FullConn(
            units=output_dim,
            activation=None,
            bias=None,
        ))

        model.summary()
        return model

    # for prediction 2 (no batchnorm)
    @staticmethod
    def cnn_lstm_no_batchnorm(input_shape, output_dim=1, num_of_conv=2, append_rnn=True):
        model = Sequential()

        # input block
        model.add(nn_utils.Conv1Dim(
            input_shape=input_shape,
            filters=32,
            kernel_size=4,
            strides=1
        ))
        model.add(Activation(tf.nn.relu))
        model.add(nn_utils.MaxPool1Dim(
            pool_size=2,
            strides=2,
        ))
        model.add(Dropout(
            rate=0.15, # 0.1,
            noise_shape=None, 
            seed=None
        ))

        # conv block
        for i in range(num_of_conv):
            model.add(nn_utils.Conv1Dim(
                input_shape=None,
                filters=64+i*32,
                kernel_size=3-i,
                strides=1
            ))
            model.add(Activation(tf.nn.relu))
            # added on 2022/12/09 for overfitting
            model.add(Dropout(
                rate=0.05,
                noise_shape=None, 
                seed=None
            ))

        # lstm block
        if append_rnn:
            model.add(Dropout(
                rate=0.1,
                noise_shape=None, 
                seed=None
            ))
            model.add(nn_utils.RnnLSTM(
                units=256,
                bias=None,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=False, 
                return_state=False
            ))
        else:
            model.add(Flatten(
                data_format=None
            ))
            model.add(nn_utils.FullConn(
                units=128,
                activation=None,
                bias=None,
            ))
            model.add(Activation(tf.nn.relu))
            model.add(Dropout(
                rate=0.2,
                noise_shape=None, 
                seed=None
            ))

        # output block
        model.add(nn_utils.FullConn(
            units=64,
            activation=None,
            bias=None,
        ))
        model.add(Activation(tf.nn.relu))
        model.add(Dropout(
            rate=0.1,
            noise_shape=None, 
            seed=None
        ))
        model.add(nn_utils.FullConn(
            units=16,
            activation=None,
            bias=None,
        ))
        model.add(Activation(tf.nn.relu))
        model.add(nn_utils.FullConn(
            units=output_dim,
            activation=None,
            bias=None,
        ))

        model.summary()
        return model

    # for prediction 2 (replace batchnorm by layernorm)
    @staticmethod
    def cnn_lstm_layernorm(input_shape, num_of_conv=1, append_rnn=True, output_dim=1):
        model = Sequential()

        # input block
        model.add(nn_utils.Conv1Dim(
            input_shape=input_shape,
            filters=32,
            kernel_size=4,
            strides=1
        ))
        model.add(nn_utils.LayerNorm(
            trainable=True
        ))
        model.add(Activation(tf.nn.relu))
        model.add(nn_utils.MaxPool1Dim(
            pool_size=2,
            strides=2,
        ))
        model.add(Dropout(
            rate=0.1,
            noise_shape=None, 
            seed=None
        ))

        # conv block
        for i in range(num_of_conv):
            model.add(nn_utils.Conv1Dim(
                input_shape=None,
                filters=64+i*32,
                kernel_size=3-i,
                strides=1
            ))
            model.add(nn_utils.LayerNorm(
                trainable=True
            ))
            model.add(Activation(tf.nn.relu))
            # added on 2022/12/09 to fix overfitting
            model.add(Dropout(
                rate=0.1, # chenged on 2022/12/09, original value is 0.05
                noise_shape=None, 
                seed=None
            ))

        # lstm block
        if append_rnn:
            model.add(Dropout(
                rate=0.1,
                noise_shape=None, 
                seed=None
            ))
            model.add(nn_utils.RnnLSTM(
                units=256,
                bias=None,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=False, 
                return_state=False
            ))
        else:
            model.add(Flatten(
                data_format=None
            ))
            model.add(nn_utils.FullConn(
                units=128,
                activation=None,
                bias=None,
            ))
            model.add(Activation(tf.nn.relu))
            model.add(Dropout(
                rate=0.2,
                noise_shape=None, 
                seed=None
            ))

        # output block
        model.add(nn_utils.FullConn(
            units=64,
            activation=None,
            bias=None,
        ))
        model.add(Activation(tf.nn.relu))
        model.add(Dropout(
            rate=0.1,
            noise_shape=None, 
            seed=None
        ))
        model.add(nn_utils.FullConn(
            units=16,
            activation=None,
            bias=None,
        ))
        model.add(Activation(tf.nn.relu))
        model.add(nn_utils.FullConn(
            units=output_dim,
            activation=None,
            bias=None,
        ))

        model.summary()
        return model

    # for prediction 3
    @staticmethod
    def _identity_block(x, filter, kernel_size=4, strides=1, conv_skip=True, norm=None):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = nn_utils.Conv1Dim(
            input_shape=None,
            filters=filter,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            resnet=True,
            restype="block"
        )(x)
        if norm and norm == "batch":
            x = nn_utils.BatchNorm(
                trainable=True
            )(x)
        elif norm and norm == "layer":
            x = nn_utils.LayerNorm(
                trainable=True
            )(x)
        elif norm:
            print(__file__, sys._getframe().f_lineno, "\n---------->", "WARNING: Invalid norm type ->", norm)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
        if norm and norm == "batch":
            pass
        elif norm and norm == "layer":
            pass
        elif norm == None:
            pass
        else:
            print(__file__, sys._getframe().f_lineno, "\n---------->", "WARNING: Invalid norm type ->", norm)
        # Layer 2
        x = nn_utils.Conv1Dim(
            input_shape=None,
            filters=filter,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            resnet=True,
            restype="block"
        )(x)
        if norm and norm == "batch":
            x = nn_utils.BatchNorm(
                trainable=True
            )(x)
        elif norm and norm == "layer":
            x = nn_utils.LayerNorm(
                trainable=True
            )(x)
        elif norm:
            print(__file__, sys._getframe().f_lineno, "\n---------->", "WARNING: Invalid norm type ->", norm)
        # Processing Residue with conv(1,1)
        if conv_skip:
            x_skip = nn_utils.Conv1Dim(
                input_shape=None,
                filters=filter,
                kernel_size=1,
                strides=1,
                padding="same",
                resnet=True,
                restype="skip"
            )(x_skip)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation('relu')(x)
        if norm and norm == "batch":
            pass
        elif norm and norm == "layer":
            pass
        elif norm == None:
            pass
        else:
            print(__file__, sys._getframe().f_lineno, "\n---------->", "WARNING: Invalid norm type ->", norm)
        return x

    # for prediction 3
    @staticmethod
    def resnet_lstm(input_shape, output_dim, block_num=2, conv_skip=True, 
        res_filters=[32, 64], kernel_size=[3, 2], strides=[1, 1], append_rnn=True, norm="batch"):

        if len(res_filters) != block_num or len(kernel_size) != block_num or len(strides) != block_num:
            print(__file__, sys._getframe().f_lineno, "\n---------->", \
                "len(res_filters) != block_num or len(kernel_size) != block_num or len(strides) != block_num")
            if min(len(res_filters), len(kernel_size), len(strides)) < block_num:
                raise KeyboardInterrupt
            else:
                raise Warning

        net_input = tf.keras.layers.Input(input_shape)
        for i in range(block_num):
            if i == 0:
                net = Models._identity_block(net_input, 
                    filter=res_filters[i], 
                    kernel_size=kernel_size[i], 
                    strides=strides[i],
                    conv_skip=conv_skip,
                    norm=norm
                )
            else:
                net = Models._identity_block(net, 
                    filter=res_filters[i], 
                    kernel_size=kernel_size[i], 
                    strides=strides[i],
                    conv_skip=conv_skip,
                    norm=norm
                )
        if norm and norm == "batch":
            net = Dropout(
                rate=0.3,
                noise_shape=None, 
                seed=None
            )(net)
        elif norm and norm == "layer":
            net = Dropout(
                rate=0.3,
                noise_shape=None, 
                seed=None
            )(net)
        elif norm == None:
            net = Dropout(
                rate=0.3,
                noise_shape=None, 
                seed=None
            )(net)
        else:
            print(__file__, sys._getframe().f_lineno, "\n---------->", "WARNING: Invalid norm type ->", norm)
        if append_rnn:
            net = nn_utils.RnnLSTM(
                units=256,
                bias=None,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=False, 
                return_state=False
            )(net)
            net = Activation(tf.nn.relu)(net)
        # output block
        net = nn_utils.FullConn(
            units=128,
            activation=None,
            bias=None,
        )(net)
        net = Activation(tf.nn.relu)(net)
        net = Dropout(
            rate=0.2,
            noise_shape=None, 
            seed=None
        )(net)
        net = nn_utils.FullConn(
            units=64,
            activation=None,
            bias=None,
        )(net)
        net = Activation(tf.nn.relu)(net)
        net = Dropout(
            rate=0.1,
            noise_shape=None, 
            seed=None
        )(net)
        net = nn_utils.FullConn(
            units=output_dim,
            activation=None,
            bias=None,
        )(net)
        model = tf.keras.models.Model(inputs=net_input, outputs=net, name="ResNet_LSTM")
        model.summary()
        return model

    # for prediction 3
    @staticmethod
    def resnet_lstm_no_batchnorm_2(input_shape, output_dim, block_num=2, conv_skip=True, 
        res_filters=[32, 64], kernel_size=[2, 2], strides=[1, 1], append_rnn=True):
        # changed on 2022/12/09 to fix underfitting
        return Models.resnet_lstm(input_shape=input_shape, output_dim=output_dim, block_num=block_num, conv_skip=conv_skip, 
            res_filters=res_filters, kernel_size=kernel_size, strides=strides, append_rnn=append_rnn, norm=None)

    # for prediction 3
    @staticmethod
    def resnet_lstm_layernorm(input_shape, output_dim, block_num=2, conv_skip=True, 
        res_filters=[32, 64], kernel_size=[2, 2], strides=[1, 1], append_rnn=True):
        return Models.resnet_lstm(input_shape=input_shape, output_dim=output_dim, block_num=block_num, conv_skip=conv_skip, 
            res_filters=res_filters, kernel_size=kernel_size, strides=strides, append_rnn=append_rnn, norm="layer")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_obj = Models(model_name="block_multi")
    params = [(50, 25), 5, None, False, True]
    model = model_obj.load_model(params)
