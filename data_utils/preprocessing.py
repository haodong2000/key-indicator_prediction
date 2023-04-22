import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm 
import time
import sys
import torch

class PreProcesser:
    
    @staticmethod
    def no_expert_knowledge(data_io):
        df = data_io.dataframe_raw
        prune_cols = []
        for col in data_io.columns_raw:
            if len(col) >= 5:
                if col[-5:-2] == "(t-" and col[-1] == ")" and "1" <= col[-2] <= "5":
                    prune_cols.append(col)
        for col in prune_cols:
            df = df.drop(col, axis=1)
        return df

    @staticmethod
    def transfer_to_sequence(x_data, y_data, device):
        size = len(x_data)
        sequence = []
        for i in range(size):
            sequence.append(np.array([x_data[i], y_data[i]]))
        return torch.FloatTensor(np.array(sequence)).to(device)

if __name__ == "__main__":
    try:
        from data_utils.get_input import Data_IO
    except:
        from get_input import Data_IO
    data_path = "./data/NSC_Si_Content_Timedelay_Data_CN.xlsx"
    data_io = Data_IO(data_path=data_path)
    pre_process = PreProcesser()
    pre_process.no_expert_knowledge(data_io)
