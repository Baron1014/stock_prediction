import requests 
import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

# stock info
u_id = "NM6101080"
password = "as987654"
scode = "2330"
vol = 5
price = 590

# Model para


class TrainSet(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 會用全0的 state
        out = self.out(r_out)
        return out
        
def read_data(stock, s_date, e_date):
    api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock, s_date, e_date)
    r = requests.get(api_url)
    history_info = json.loads(r.text) 
    # data rerset low_price to next day
    target = [data["low"] for i, data in enumerate(history_info) if i != 0]
    history_training = [
        [data['capacity'], data['turnover'], data["open"], data["high"],data["close"],data["change"],data["transaction_volume"], target[i]] 
        for i, data in enumerate(history_info) if i != len(history_info)-1
    ]
    # normalize x: input type "numpy array"
    history_array = np.array(history_training)
    y = history_array[:, -1].reshape(-1, 1)
    nor_x = normalize(history_array[:, :-1])

    return torch.Tensor(np.concatenate((nor_x, y), axis=1))

def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return  (data - mean) / std

def main():
    r = requests.post("http://140.116.86.242:8081/stock/api/v1/buy", data={"uname":u_id, "pass":password, "scode": scode, "svol": str(vol), "sell_price":str(price)})
    print(r)


if __name__ == "__main__":
    test_data = read_data(2330, 20150301, 20220322)
    print(test_data.shape)
    print(test_data[0])
    train = TrainSet(test_data)
    print(train)
    mydataloader = DataLoader(dataset=train, batch_size=64)

