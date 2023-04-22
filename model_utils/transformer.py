import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0).transpose(0, 1) # [5000, 1, d_model],so need seq-len <= 5000
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe[:x.size(0), :].repeat(1,x.shape[1],1).shape ,'---',x.shape)
        # dimension 1 maybe inequal batchsize
        return x + self.pe[:x.size(0), :].repeat(1,x.shape[1],1)
          
class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_of_var=6, select_dim=1, num_layers=1, nhead=10, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.fuse_dim = nn.Linear(num_of_var, 1)
        self.input_embedding  = nn.Linear(select_dim, feature_size)
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # self.decoder_embedding  = nn.Linear(1, feature_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_embedidng = nn.Linear(feature_size, select_dim)
        self.un_fuse_dim = nn.Linear(1, num_of_var)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.output_embedidng.bias.data.zero_()
        self.output_embedidng.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src with shape (input_window, batch_len, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        # src = self.fuse_dim(src)
        # src = torch.squeeze(src, dim=-1)
        src = self.input_embedding(src) # linear transformation before positional embedding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.transformer_decoder(output, src)
        output = self.output_embedidng(output)
        # output = torch.unsqueeze(output, dim=-1)
        # output = self.un_fuse_dim(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TransUtils():

    @staticmethod
    def get_batch(input_data, i, batch_size):
        batch_len = min(batch_size, len(input_data) - i)
        data = input_data[i:i + batch_len]
        input = data[:, 0, :]
        target = data[:, 1, :]
        return input, target

    @staticmethod
    def train(model, train_data, batch_size, optimizer, loss_criterion, accuracy_criterion):
        model.train() # Turn on the train mode \o/
        total_loss = []
        total_acc = []

        for batch, i in enumerate(range(0, len(train_data), batch_size)):  # Now len-1 is not necessary
            data, targets = TransUtils.get_batch(train_data, i , batch_size)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_criterion(output, targets)
            acc = accuracy_criterion(output, targets)
            loss.backward()

            optimizer.step()

            total_loss.append(loss.item())
            total_acc.append(acc.item())
        
        return np.mean(total_loss), np.mean(total_acc)
    
    @staticmethod
    def validate(eval_model, data_source, batch_size, loss_criterion, accuracy_criterion):
        eval_model.eval() 
        total_loss = []
        total_acc = []
        total_out = []
        with torch.no_grad():
            for batch, i in enumerate(range(0, len(data_source), batch_size)):
                data, targets = TransUtils.get_batch(data_source, i, batch_size)
                output = eval_model(data)
                loss = loss_criterion(output, targets)
                acc = accuracy_criterion(output, targets)          
                total_loss.append(loss.item())
                total_acc.append(acc.item())
                total_out += list(output[:,:,-1].cpu().numpy())
        return np.mean(total_loss), np.mean(total_acc), np.array(total_out)

