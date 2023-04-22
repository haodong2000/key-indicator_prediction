import torch
import torch.nn as nn
import numpy as np
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class cl_utils():
    @staticmethod
    def get_batch(input_data, i, batch_size):
        batch_len = min(batch_size, len(input_data[0]) - i)
        input = input_data[0][i:i + batch_len]
        target = input_data[1][i:i + batch_len]
        return torch.FloatTensor(input).to(device), torch.FloatTensor(target).to(device)

    @staticmethod
    def train(model, train_data, memory, batch_size, optimizer, loss_criterion, accuracy_criterion):
        model.train() # Turn on the train mode \o/
        total_loss = []
        total_acc = []

        for batch, i in enumerate(range(0, len(train_data[0]), batch_size)):
            data, targets = cl_utils.get_batch(train_data, i , batch_size)
            memory.write(data)
            optimizer.zero_grad()
            output = model(data, memory.read())
            # output = model(data)
            loss = loss_criterion(output, targets)
            acc = accuracy_criterion(output, targets)
            loss.backward()

            optimizer.step()

            total_loss.append(loss.item())
            total_acc.append(acc.item())

        return np.mean(total_loss), np.mean(total_acc)

    @staticmethod
    def validate(eval_model, data_source, memory, batch_size, loss_criterion, accuracy_criterion):
        eval_model.eval() 
        total_loss = []
        total_acc = []
        total_out = []
        with torch.no_grad():
            for batch, i in enumerate(range(0, len(data_source[0]), batch_size)):
                data, targets = cl_utils.get_batch(data_source, i, batch_size)
                memory.write(data)
                output = eval_model(data, memory.read())
                # output = eval_model(data)
                loss = loss_criterion(output, targets)
                acc = accuracy_criterion(output, targets)
                total_loss.append(loss.item())
                total_acc.append(acc.item())
                total_out += list(output.cpu().numpy())
        return np.mean(total_loss), np.mean(total_acc), np.array(total_out)

class memory():
    def __init__(self, max_length, channel) -> None:
        self.max_length = max_length
        self.data = None
        self.batch_size = 0
        self.channel = channel

    def read(self):
        if self.data is None:
            return torch.zeros((self.batch_size, self.max_length, self.channel)).to(device)
        memory_data = []
        if self.data.shape[0] > self.max_length + self.batch_size:
            for i in range(self.batch_size):
                if i == 0:
                    memory_data.append(np.array(fix_data[-self.max_length:].cpu()))
                else:
                    memory_data.append(np.array(self.data[-self.max_length - i:-i].cpu()))
        else:
            prefix = torch.zeros((self.max_length + self.batch_size - self.data.shape[0], self.data.shape[1])).to(device)
            fix_data = torch.cat((prefix, self.data), dim=0)
            for i in range(self.batch_size):
                if i == 0:
                    memory_data.append(np.array(fix_data[-self.max_length:].cpu()))
                else:
                    memory_data.append(np.array(fix_data[-self.max_length - i:-i].cpu()))
        return torch.FloatTensor(np.array(memory_data)).to(device)

    def write(self, new_data):
        self.batch_size = new_data.shape[0]
        if self.data is None:
            self.data = torch.cat((new_data[0], 
                                  torch.FloatTensor(np.array([item[-1] for item in new_data[1:].cpu().numpy()])).to(device)), dim=0)
        else:
            self.data = torch.cat((self.data, 
                                   torch.FloatTensor(np.array([item[-1] for item in new_data.cpu().numpy()])).to(device)), dim=0)
        if self.data.shape[0] > self.max_length + self.batch_size:
            self.data = self.data[-self.max_length:]
    
    def __len__(self):
        return self.data.shape[0]

class mlp_end(nn.Module):
    def __init__(self, input_dim=128, output_dim=1):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.05)
        self.dense2 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        return x

class fast_lstm(nn.Module):
    def __init__(self, output_dim=128, input_dim=512, num_lstm_layers=2, bias=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 256, num_layers=num_lstm_layers, batch_first=True, bias=bias)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(256, 128)
        # torch.nn.init.zeros_(self.dense1.bias)
        self.dense2 = nn.Linear(128, output_dim)
        # torch.nn.init.zeros_(self.dense2.bias)
        self.dropout1 = nn.Dropout(0.2)


    def forward(self, x):
        # for i in range(len(x)):
        #     torch.reshape(x[i], x[i].T.shape)
        x, _ = self.lstm(x)
        x = self.relu(x[:, -1, :])
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        return x

class slow_lstm(nn.Module):
    def __init__(self, output_dim=128, input_dim=512, num_lstm_layers=2, bias=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, num_layers=num_lstm_layers, batch_first=True, bias=bias)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(128, 64)
        # torch.nn.init.zeros_(self.dense1.bias)
        self.dense2 = nn.Linear(64, output_dim)
        # torch.nn.init.zeros_(self.dense2.bias)
        self.dropout1 = nn.Dropout(0.1)
        

    def forward(self, x):
        # for i in range(len(x)):
        #     torch.reshape(x[i], x[i].T.shape)
        x, _ = self.lstm(x)
        x = self.relu(x[:, -1, :])
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        return x

class fastnet_attention_resnet(nn.Module):
    def __init__(self, seq_length=500, input_dim=6, output_dim=256, feature_size=512, 
                 num_heads=16, dropout=0.1, num_layers=2):
        super().__init__()
        self.feature_size = feature_size
        self.input_embedding = nn.Linear(seq_length, feature_size)
        self.multihead_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        # add resnet layers
        self.resblock1 = nn.Sequential(
            nn.Conv1d(input_dim, 8, kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=2, stride=1, padding=0),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.resblock2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=2, stride=1, padding=0),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.conv1 = nn.Conv1d(input_dim, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        torch.nn.init.ones_(self.conv1.weight)
        torch.nn.init.ones_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)

        for layer in self.resblock1:
            if isinstance(layer, nn.Conv1d):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        for layer in self.resblock2:
            if isinstance(layer, nn.Conv1d):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        self.output_embedding = nn.Linear(feature_size, output_dim)

    def forward(self, x):
        # x = self.input_embedding(x)
        x, _ = self.multihead_attention(x, x, x)
        x = x.permute(0, 2, 1)
        x0 = self.conv1(x)
        x1 = self.resblock1(x)
        x = x0 + x1
        x2 = self.conv2(x)
        x3 = self.resblock2(x)
        x = x2 + x3
        # x = self.output_embedding(x)
        x = x.permute(0, 2, 1)
        return x

class slownet_attention_resnet(nn.Module):
    def __init__(self, seq_length_long=500*10, seq_length=500, input_dim=6, output_dim=256, feature_size=512, 
                 num_heads=16, dropout=0.1, num_layers=2):
        super().__init__()
        self.feature_size = feature_size
        self.input_embedding = nn.Linear(seq_length_long, seq_length)
        self.multihead_attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        # add resnet layers
        self.resblock1 = nn.Sequential(
            nn.Conv1d(input_dim, 8, kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=2, stride=1, padding=0),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.resblock2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=2, stride=1, padding=0),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.conv1 = nn.Conv1d(input_dim, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1)
        torch.nn.init.ones_(self.conv1.weight)
        torch.nn.init.ones_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)

        for layer in self.resblock1:
            if isinstance(layer, nn.Conv1d):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        for layer in self.resblock2:
            if isinstance(layer, nn.Conv1d):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        self.output_embedding = nn.Linear(feature_size, output_dim)

    def forward(self, x):
        x, _ = self.multihead_attention(x, x, x)
        x = x.permute(0, 2, 1)
        x = self.input_embedding(x)
        x0 = self.conv1(x)
        x1 = self.resblock1(x)
        x = x0 + x1
        x2 = self.conv2(x)
        x3 = self.resblock2(x)
        x = x2 + x3
        # x = self.output_embedding(x)
        x = x.permute(0, 2, 1)
        return x

class continual_learning(nn.Module):

    def __init__(self, seq_length=500, seq_length_long=500*50, output_dim_1=256, output_dim_2=64, 
                 output_dim=1, input_dim_new=6, feature_size=512, num_heads=16, dropout=0.1, num_layers=2) -> None:
        super().__init__()
        self.fastnet_1 = fastnet_attention_resnet(seq_length=seq_length, input_dim=output_dim,
                                                  output_dim=output_dim_1, feature_size=feature_size, 
                                                  num_heads=num_heads, dropout=dropout, 
                                                  num_layers=num_layers)
        self.slownet_1 = slownet_attention_resnet(seq_length_long=seq_length_long,
                                                  seq_length=seq_length, input_dim=output_dim, 
                                                  output_dim=int(output_dim_1/2), feature_size=feature_size, 
                                                  num_heads=num_heads, dropout=dropout, 
                                                  num_layers=num_layers)
        self.fastnet_2 = fast_lstm(output_dim=output_dim_2, input_dim=output_dim_1 + int(output_dim_1/2), 
                                   num_lstm_layers=num_layers)
        self.slownet_2 = slow_lstm(output_dim=int(output_dim_2/2), input_dim=int(output_dim_1/2), 
                                   num_lstm_layers=num_layers)
        self.mlp_end = mlp_end(input_dim=output_dim_2 + int(output_dim_2/2), output_dim=output_dim)

    def forward(self, x, memory):
        x_fast_1 = self.fastnet_1(x)
        x_slow_1 = self.slownet_1(memory)
        x_1 = torch.cat((x_slow_1, x_fast_1), dim=-1)
        x_fast_2 = self.fastnet_2(x_1)
        x_slow_2 = self.slownet_2(x_slow_1)
        x_2 = torch.cat((x_slow_2, x_fast_2), dim=-1)
        x = self.mlp_end(x_2)
        return x

if __name__ == "__main__":
    # print(Simple_LSTM(500, 1, 1))
    x = torch.rand(9, 50, 6).to('cpu')
    memory = torch.zeros(size=(9, 500, 6)).to('cpu')
    f = continual_learning(seq_length=50, seq_length_long=500, output_dim_1=32, output_dim_2=32, 
            output_dim=6, input_dim_new=6, feature_size=512, num_heads=1, dropout=0.1, num_layers=2).to('cpu')
    print(f)
    y = f(x, memory)
    print(y.shape)
