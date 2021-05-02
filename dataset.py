import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence
from torchvision import transforms


class MyData(Dataset):
    def __init__(self, datax, datay,transform=None):
        self.datax = datax
        self.datay = datay
        self.transform = transform

    def __len__(self):
        return len(self.datax)

    def __getitem__(self, idx):
        if(self.transform != None):
            return self.transform(self.datax[idx]),self.transform(self.datay[idx])
        return self.datax[idx],self.datay[idx]

def collate_fn(data):
    # data.squesze(1)
    traindata = data
    data = []
    label = []
    for (x,y) in traindata:
        data.append(x)
        label.append(y)
    data.sort(key=lambda x: len(x), reverse=True)
    seq_len = [s.size(0) for s in data]
    data = pad_sequence(data, batch_first=True).float()    
    # data = data.unsqueeze(-1)
    data = pack_padded_sequence(data, seq_len, batch_first=True)
    return data,label

def test():
    a = torch.tensor([1,2,3,4])
    b = torch.tensor([5,6,7])
    c = torch.tensor([7,8])
    d = torch.tensor([9])
    train_x = [a, b, c, d]

    X = []
    Y = []
    true_file = './testdata/000001/true'
    false_file = './testdata/000001/false'
    for root, dirs, files in os.walk(true_file):
        for file in files:
            # print(file)
            df = pd.read_csv(os.path.join(true_file,file),index_col=0).iloc[:,2:10]
            df.drop('pre_close',axis=1,inplace=True)
            # df = df.apply(lambda x : (x-min(x)) / (max(x) - min(x)))
            X.append(torch.from_numpy(np.array(df.values, dtype=np.float32)))
            Y.append(torch.from_numpy(np.array(1, dtype=np.float32)))
            print(df.values)
        # assert 0
    print(X)
    print(Y)
    train_x = X
    train_y = Y
    data = MyData(train_x,train_y)
    test = data[0]
    data_loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate_fn)
    batch_x = iter(data_loader).next()
    (a,b)=batch_x
    rnn = nn.LSTM(7, 4, 1, batch_first=True)
    # h0 (num_layer,batchsize,hidden_size),和 input_size无关
    # offical doc :(num_layers * num_directions, batch, hidden_size)
    h0 = torch.rand(1, 2, 4).float()
    c0 = torch.rand(1, 2, 4).float()
    out, (h1, c1) = rnn(a, (h0, c0))
def getData(corpusFile,is_cache=False,label: str = ['train','valid']):
# def getData(corpusFile,is_cache=False,label: str = 'train'):
    X = []
    Y = []
    print(label)
    if(is_cache):
        with open('dataX.pkl','rb') as f:
            X = pickle.load(f)
        with open('dataY.pkl','rb') as f:
            Y = pickle.load(f)
    else:
        for root ,dirs ,files in os.walk(corpusFile):
            print(dirs)
            break
        for myfiles in dirs:
            true_file = os.path.join(corpusFile,myfiles,'true')
            false_file = os.path.join(corpusFile,myfiles,'false')
            for root, dirs, files in os.walk(true_file):
                    for file in files:
                        # print(file)
                        df = pd.read_csv(os.path.join(true_file,file),index_col=0).iloc[:,2:10]
                        df.drop('pre_close',axis=1,inplace=True)
                        df = df.apply(lambda x : (x-min(x)) / (max(x) - min(x)))
                        X.append(torch.from_numpy(np.array(df.values, dtype=np.float32)))
                        Y.append(torch.from_numpy(np.array(1, dtype=np.float32)))
                        print(df.values)
            for root, dirs, files in os.walk(false_file):
                    for file in files:
                        # print(file)
                        df = pd.read_csv(os.path.join(false_file,file),index_col=0).iloc[:,2:10]
                        df.drop('pre_close',axis=1,inplace=True)
                        df = df.apply(lambda x : (x-min(x)) / (max(x) - min(x)))
                        X.append(torch.from_numpy(np.array(df.values, dtype=np.float32)))
                        Y.append(torch.from_numpy(np.array(0, dtype=np.float32)))
                        print(df.values)
        with open('dataX.pkl','wb') as f:
            pickle.dump(X, f)
        with open('dataY.pkl','wb') as f:
            pickle.dump(Y, f)
    train_x = X
    train_y = Y
    data = MyData(train_x,train_y)
    data_loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate_fn)
    # batch_x = iter(data_loader).next()
    # (a,b)=batch_x
    # rnn = nn.LSTM(7, 4, 1, batch_first=True)
    return data_loader

getData('./testdata',True)
# print(getData.__annotations__)