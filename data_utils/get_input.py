"""
1. what to predict?
    - given data[0:i][:], output data[i+1:i+1+k][:]
        - data_hat[i+1] = F(data[0], data[1], ..., data[i])
        - data_hat[i+2] = F(data[1], data[2], ..., data_hat[i+1])
        - ...
        - data_hat[i+1+k] = F(data[k-1], data[k], ..., data_hat[i+1], ..., data_hat[i+k])
    - given data[0:i][:], output data[i+1+delay:i+1+delay+k][:]
    - given data[i][0:j], output data[i][j+1:j+1+k]
    - given data[i][0:j], output data[i][j+1+delay:j+1+delay+k]
        - i, j, n could be defined by percentage (or window size)
        - k is the number of steps
2. or some novel ideas
    - given discrete data instead of sequential data
        - this could be implemented in _train_test_split()
"""

import numpy as np
import os
import pandas as pd
from tqdm import tqdm 
import time
import random
import sys
try:
    from data_utils.preprocessing import PreProcesser
except:
    from preprocessing import PreProcesser
from sklearn import preprocessing

class Data_IO:

    def __init__(self, data_path):
        self.data_path = data_path
        start_time = time.time()
        print(__file__, sys._getframe().f_lineno, "\n---------->", "start loading " + data_path + " ...")
        self.dataframe_raw = pd.read_excel(self.data_path)
        self.columns_raw = self.dataframe_raw.columns
        self.dataframe_raw = self._data_preprocess(self.dataframe_raw)
        self.dataframe = PreProcesser.no_expert_knowledge(self)
        self.columns = self.dataframe.columns
        print(__file__, sys._getframe().f_lineno, "\n---------->", "loading finished in", time.time() - start_time, "seconds")
        del start_time

    @staticmethod
    def _data_preprocess(dataframe):
        # normalization and de-noising
        columns = dataframe.columns
        data_val = dataframe.values
        min_max_scaler = preprocessing.MinMaxScaler()
        data_val_scaled = min_max_scaler.fit_transform(data_val)
        dataframe_nor = pd.DataFrame(data_val_scaled)
        dataframe_nor.columns = columns
        del columns, data_val, min_max_scaler, data_val_scaled
        return dataframe_nor

    @staticmethod
    def _train_test_split(x, y, split_ratio, shuffle=True):
        assert len(x) == len(y)
        num_train = round(split_ratio * len(x))
        del split_ratio

        def _shuffle_data(a, b):
            assert len(a) == len(b)
            start_state = random.getstate()
            random.shuffle(a)
            random.setstate(start_state)
            random.shuffle(b)
            return a, b
            pass
        
        if shuffle: x, y = _shuffle_data(x, y)
        return np.array(x[:num_train]), np.array(y[:num_train]), np.array(x[num_train:]), np.array(y[num_train:])

    def io_soft_pd_xy(self, select_dim, split_ratio, shuffle):
        output_dim = select_dim
        input_dim = []
        for i in range(self.columns.size):
            if i not in output_dim:
                input_dim.append(i)
        input_dim = self.dataframe.columns[input_dim]
        output_dim = self.dataframe.columns[output_dim]
        x = self.dataframe[input_dim].values
        y = self.dataframe[output_dim].values
        del input_dim, output_dim
        return self._train_test_split(x, y, split_ratio=split_ratio, shuffle=shuffle)

    def io_soft_window_xy(self, select_dim, window_size, num_of_step, delay, split_ratio, shuffle):
        output_dim = select_dim
        input_dim = []
        for i in range(self.columns.size):
            if i not in output_dim:
                input_dim.append(i)
        input_dim = self.dataframe.columns[input_dim]
        output_dim = self.dataframe.columns[output_dim]
        input_data = self.dataframe[input_dim].values
        output_data = self.dataframe[output_dim].values

        data_len = len(output_data)
        x, y, idx = [], [], 0

        print(__file__, sys._getframe().f_lineno, "\n---------->", "generating x_train, y_train, x_test, y_test ...")
        for _ in tqdm(output_data, total=data_len):
            if idx < window_size:
                pass
            elif idx <= data_len - delay - num_of_step:
                x.append(input_data[idx - window_size:idx])
                y.append(output_data[idx + delay:idx + delay + num_of_step])
            else:
                break
            idx += 1
        
        del input_dim, output_dim, select_dim, input_data, output_data, window_size, num_of_step, delay, data_len, idx
        return self._train_test_split(x, y, split_ratio=split_ratio, shuffle=shuffle)

    def io_window_xy(self, select_dim, window_size, num_of_step, delay, split_ratio, shuffle):
        select_dim = self.dataframe.columns[select_dim]
        select_data = self.dataframe[select_dim].values
        data_len = len(select_data)
        x, y, idx = [], [], 0

        print(__file__, sys._getframe().f_lineno, "\n---------->", "generating x_train, y_train, x_test, y_test ...")
        for _ in tqdm(select_data, total=data_len):
            if idx < window_size:
                pass
            elif idx <= data_len - delay - num_of_step:
                x.append(select_data[idx - window_size:idx])
                y.append(select_data[idx + delay:idx + delay + num_of_step])
            else:
                break
            idx += 1
        
        del select_dim, select_data, window_size, num_of_step, delay, data_len, idx
        return self._train_test_split(x, y, split_ratio=split_ratio, shuffle=shuffle)

    def io_window_xy_CL(self, select_dim, window_size, num_of_step, delay, split_ratio, shuffle):
        select_dim = self.dataframe.columns[select_dim]
        select_data = self.dataframe[select_dim].values
        data_len = len(select_data)
        x, y, idx = [], [], 0

        print(__file__, sys._getframe().f_lineno, "\n---------->", "generating x_train, y_train, x_test, y_test ...")
        for _ in tqdm(select_data, total=data_len):
            if idx < window_size:
                pass
            elif idx <= data_len - delay - num_of_step:
                x.append(select_data[idx - window_size:idx].T)
                y.append(select_data[idx + delay:idx + delay + num_of_step].T)
            else:
                break
            idx += 1
        
        del select_dim, select_data, window_size, num_of_step, delay, data_len, idx
        return self._train_test_split(x, y, split_ratio=split_ratio, shuffle=shuffle)

    def io_window_xy_CL_soft(self, select_dim, window_size, num_of_step, delay, split_ratio, shuffle):
        size = len(self.dataframe.columns)
        all_dims = [i for i in range(size)]
        for dim in select_dim:
            all_dims.remove(dim)
        select_dim_x = all_dims

        select_dim_x = self.dataframe.columns[select_dim_x]
        select_dim = self.dataframe.columns[select_dim]
        select_data_x = self.dataframe[select_dim_x].values
        select_data = self.dataframe[select_dim].values
        data_len = len(select_data)
        x, y, idx = [], [], 0

        print(__file__, sys._getframe().f_lineno, "\n---------->", "generating x_train, y_train, x_test, y_test ...")
        for _ in tqdm(select_data, total=data_len):
            if idx < window_size:
                pass
            elif idx <= data_len - delay - num_of_step:
                x.append(select_data_x[idx - window_size:idx].T)
                y.append(select_data[idx + delay:idx + delay + num_of_step].T)
            else:
                break
            idx += 1
        
        del select_dim, select_data, window_size, num_of_step, delay, data_len, idx
        return self._train_test_split(x, y, split_ratio=split_ratio, shuffle=shuffle)

    def io_window_xy_CL_all(self, select_dim, window_size, num_of_step, delay, split_ratio, shuffle):
        select_dim_x = self.dataframe.columns
        select_data_x = self.dataframe[select_dim_x].values
        select_dim = self.dataframe.columns[select_dim]
        select_data = self.dataframe[select_dim].values
        data_len = len(select_data)
        x, y, idx = [], [], 0

        print(__file__, sys._getframe().f_lineno, "\n---------->", "generating x_train, y_train, x_test, y_test ...")
        for _ in tqdm(select_data, total=data_len):
            if idx < window_size:
                pass
            elif idx <= data_len - delay - num_of_step:
                x.append(select_data_x[idx - window_size:idx].T)
                y.append(select_data[idx + delay:idx + delay + num_of_step].T)
            else:
                break
            idx += 1
        
        del select_dim, select_data, window_size, num_of_step, delay, data_len, idx
        return self._train_test_split(x, y, split_ratio=split_ratio, shuffle=shuffle)

    def io_sequence_xy(self, select_dim, window_size, num_of_step, delay, split_ratio, shuffle):
        select_dim = self.dataframe.columns[select_dim]
        select_data = self.dataframe[select_dim].values
        data_len = len(select_data)
        x, y, idx = [], [], 0

        print(__file__, sys._getframe().f_lineno, "\n---------->", "generating x_train, y_train, x_test, y_test ...")
        for _ in tqdm(select_data, total=data_len):
            if idx < window_size:
                pass
            elif idx <= data_len - delay - num_of_step:
                x.append(np.array(select_data[idx - window_size:idx]).T)
                y.append(np.array(select_data[idx - window_size + num_of_step:idx + num_of_step]).T)
            else:
                break
            idx += 1
        
        del select_dim, select_data, window_size, num_of_step, delay, data_len, idx
        return self._train_test_split(x, y, split_ratio=split_ratio, shuffle=shuffle)

    def io_full_xy(self, select_dim, split_ratio, shuffle):
        select_dim = self.dataframe.columns[select_dim]
        select_data = self.dataframe[select_dim].values
        data_len = len(select_data)
        x, y = [[i] for i in range(data_len)], select_data
        del select_data, data_len
        return self._train_test_split(np.array(x), y, split_ratio=split_ratio, shuffle=shuffle)

if __name__ == "__main__":
    data_path = "./data/NSC_Si_Content_Timedelay_Data_CN.xlsx"
    data_io = Data_IO(data_path=data_path)
    select_dim = [0] + [52 + i for i in range(5)]
    
    x_train_1, y_train_1, x_test_1, y_test_1 = \
        data_io.io_window_xy(select_dim=select_dim, \
            window_size=100, num_of_step=1, delay=0, split_ratio=0.7, shuffle=False)
    print(__file__, sys._getframe().f_lineno, "\n---------->", \
        x_train_1.shape, y_train_1.shape, x_test_1.shape, y_test_1.shape)

    x_train_4, y_train_4, x_test_4, y_test_4 = \
        data_io.io_soft_pd_xy(select_dim=select_dim, split_ratio=0.7, shuffle=False)
    print(__file__, sys._getframe().f_lineno, "\n---------->", \
        x_train_4.shape, y_train_4.shape, x_test_4.shape, y_test_4.shape)
