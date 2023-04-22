import torch
import torch.nn as nn
import config
import tensorflow as tf

class LossFunction():
    @staticmethod
    # root mean squared error (rmse) for regression (only for Keras tensors)
    def rmse(y_true, y_pred):
        from keras import backend
        return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    @staticmethod
    # mean squared error (mse) for regression  (only for Keras tensors)
    def mse(y_true, y_pred):
        from keras import backend
        return backend.mean(backend.square(y_pred - y_true), axis=-1)

    @staticmethod
    def adj_r_square(y_true, y_pred, adjested=True, \
        window_size=config.global_params["window_size"], \
            num_of_var=config.global_params["select_dim"]):
        from keras import backend as K
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        r2 = (1 - SS_res/(SS_tot + K.epsilon()))
        if adjested:
            return (1 - (((window_size - 1)/(window_size - num_of_var - 1)) * (1 - r2)))
        else:
            return r2

    @staticmethod
    # coefficient of determination (R^2) for regression  (only for Keras tensors)
    def r_square(y_true, y_pred):
        from keras import backend as K
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return (1 - SS_res/(SS_tot + K.epsilon()))

    @staticmethod
    def rmse_torch(y_pred, y_true):
        y_pred, y_true = y_pred[:, :, -1], y_true[:, :, -1]
        return torch.sqrt(torch.mean((y_pred - y_true)**2))

    @staticmethod
    def rmse_torch_CL(y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true)**2))

    @staticmethod
    def r_square_torch_CL(y_pred, y_true):
        from keras import backend as K
        SS_tot =  torch.sum(torch.square(y_true - y_pred)) 
        SS_res = torch.sum(torch.square(y_true - torch.mean(y_true))) 
        return (1 - SS_tot/(SS_res + K.epsilon()))

    @staticmethod
    def r_square_torch(y_pred, y_true):
        y_pred, y_true = y_pred[:, :, -1], y_true[:, :, -1]
        from keras import backend as K
        SS_tot =  torch.sum(torch.square(y_true - y_pred)) 
        SS_res = torch.sum(torch.square(y_true - torch.mean(y_true))) 
        return (1 - SS_tot/(SS_res + K.epsilon()))


if __name__ == '__main__':
    a, b = tf.convert_to_tensor([[1], [2], [3]]), tf.convert_to_tensor([[1], [2], [3]])
    loss = LossFunction.r_square(a, b)
    print(loss)
