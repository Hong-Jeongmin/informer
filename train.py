
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from datetime import timedelta
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import argparse, pickle

# model을 import
from models.model import Informer
device = torch.device('cuda:0')
class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean
        
# 시간 특징을 freq에 따라 추출
def time_features(dates, freq='h'):
    dates['month'] = dates.date.apply(lambda row:row.month,1)
    dates['day'] = dates.date.apply(lambda row:row.day,1)
    dates['weekday'] = dates.date.apply(lambda row:row.weekday(),1)
    dates['hour'] = dates.date.apply(lambda row:row.hour,1)
    dates['minute'] = dates.date.apply(lambda row:row.minute,1)
    dates['minute'] = dates.minute.map(lambda x:x//15)
    freq_map = {
        'y':[],'m':['month'],'w':['month'],'d':['month','day','weekday'],
        'b':['month','day','weekday'],'h':['month','day','weekday','hour'],
        't':['month','day','weekday','hour','minute'],
    }
    return dates[freq_map[freq.lower()]].values

# 한번의 batch를 실행하는 코드
def _process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float()
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
    dec_inp = torch.cat([batch_y[:,:label_len,:], dec_inp], dim=1).float().to(device)
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    batch_y = batch_y[:,-pred_len:,0:].to(device)
    return outputs, batch_y

class Dataset_Pred(Dataset):
    def __init__(self, dataframe, size=None, scale=True, freq='h'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.dataframe = dataframe
        
        self.scale = scale
        self.freq = freq
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = self.dataframe
        df_raw["date"] = pd.to_datetime(df_raw["date"])

        delta = df_raw["date"].iloc[1] - df_raw["date"].iloc[0]

        border1 = 0
        border2 = len(df_raw)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]


        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

def nmae(y_pred, y_true, reduction='mean'):
    if reduction == 'mean':
        return torch.mean(torch.abs(y_pred - y_true)/y_true)
    elif reduction == 'sum':
        return torch.sum(torch.abs(y_pred - y_true)/y_true)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--company', default='A', type=str)
    parser.add_argument('--pred_len', default=24*30, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=int)
    args = parser.parse_args()
    pred_len = args.pred_len     
    seq_len=args.pred_len
    label_len=args.pred_len
    print("Company : ", args.company)


    data= pd.read_csv(f'data/gas_{args.company}.csv')
    data.rename(columns = {'공급량' : 'value'}, inplace = True)
    min_max_scaler = MinMaxScaler()
    data["value"] = min_max_scaler.fit_transform(data["value"].to_numpy().reshape(-1, 1)).reshape(-1)
    data.head()
    data_train = data.copy()
    # data_train = data.iloc[:-24*365].copy()
    # data_valid = data.iloc[-24*365:].copy()

    seq_len = pred_len#인풋 크기
    label_len = pred_len#디코더에서 참고할 크기
    pred_len = pred_len#예측할 크기

    batch_size = 32
    shuffle_flag = True
    num_workers = 0
    drop_last = True

    train_dataset = Dataset_Pred(dataframe=data_train ,scale=True, size = (seq_len, label_len,pred_len))
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle_flag,num_workers=num_workers,drop_last=drop_last)
    # valid_dataset = Dataset_Pred(dataframe=data_valid ,scale=True, size = (seq_len, label_len,pred_len))
    # valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,drop_last=drop_last)

    enc_in = 1
    dec_in = 1
    c_out = 1

    model = Informer(enc_in, dec_in, c_out, seq_len, label_len, pred_len, device = device).to(device)

    model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optim, 'min', patience=3)
    scheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=15, gamma=0.1)
    
    from utils.tools import EarlyStopping
    train_epochs = 30
    early_stopping = EarlyStopping(patience=5, verbose=True)
    for epoch in range(train_epochs):
        total_loss, total_num = 0, 0
        model.train()
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(tqdm(train_loader, leave=False)):
            model_optim.zero_grad()
            pred, true = _process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = nmae(pred, true)
            total_loss += loss.item()
            total_num += pred.size(0)
            loss.backward()
            model_optim.step()
        train_loss = total_loss/total_num
        
        
        ##validation##
        # total_loss, total_num = 0, 0
        # model.eval()
        # with torch.no_grad():
        #     for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(valid_loader):
        #         pred, true = _process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
        #         loss = nmae(pred, true)
        #         total_loss += loss.item()
        #         total_num += pred.size(0)
        #     valid_loss = total_loss/total_num
        #     scheduler.step(valid_loss)
        # early_stopping(valid_loss, model, f'model{args.company}.pth')

        # print(f"{epoch+1}/{train_epochs} Train : {train_loss:.4f}  Valid : {valid_loss:.4f}")
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        if epoch%10==9:
            torch.save(model.state_dict(), f'./model{args.company}_{epoch}.pth')