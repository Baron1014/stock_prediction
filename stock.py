import requests 
import json
import torch
import wandb
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse

# stock info
u_id = "NM6101080"
password = "as987654"
scode = "2330"
vol = 5
price = 590

# Model para
learning_rate = 0.0001
EPOCH = 8000


class MydataSet(Dataset):
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

class ReadData():
    def __init__(self, stock, s_date, e_date):
        api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock, s_date, e_date)
        r = requests.get(api_url)
        history_info = json.loads(r.text) 
        # data rerset low_price to next day
        target = [data["low"] for i, data in enumerate(history_info) if i != 0]
        history_data = [
            [data['capacity'], data['turnover'], data["open"], data["high"],data["close"],data["change"],data["transaction_volume"], target[i]] 
            for i, data in enumerate(history_info) if i != len(history_info)-1
        ]
        history_data = np.array(history_data)
        self.__training_data, self.__testing_data = train_test_split(history_data, test_size=0.2, shuffle=False)
        
    @property
    def get_training_data(self):
        # normalize x: input type "numpy array"
        y = self.__training_data[:, -1].reshape(-1, 1)
        nor_x = self.normalize(self.__training_data[:, :-1])

        return torch.Tensor(np.concatenate((nor_x, y), axis=1))
    
    @property
    def get_testing_data(self):
        # normalize x: input type "numpy array"
        y = self.__testing_data[:, -1].reshape(-1, 1)
        nor_x = self.normalize(self.__testing_data[:, :-1])

        return torch.Tensor(nor_x), y

    def normalize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return  (data - mean) / std

def train(training_data, feature_number):
    wandb.init(project='Stock', entity="baron")
    train_data = MydataSet(training_data)
    trainloader = DataLoader(dataset=train_data, batch_size=64)
    model = RNN(feature_number)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimize all cnn parameters
    loss_func = nn.MSELoss()

    for step in range(EPOCH):
        for tx, ty in trainloader:
            output = model(torch.unsqueeze(tx, dim=0))
            loss = loss_func(torch.squeeze(output), ty)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            wandb.log({"training_loss": loss}, step=step)
            optimizer.step()
        if step % 500 ==0:
            print("[{}/{}] Loss:{:.4f}".format(step, EPOCH, loss.item()))

    torch.save(model.state_dict(), "./models/save_model.pth")
    return model

def predict(model, test_x, test_y):
    predict_y = model(test_x)


def load_model(model_name, feature_number):
    device = torch.device('cpu')
    model = RNN(feature_number)
    model.load_state_dict(torch.load("models/{}.pth".format(model_name), map_location=device))

    return model

def evaluation(model, test_x, test_y):
    predict_y = model(test_x)
    predict_y_array = predict_y.cpu().detach().numpy()

    # low price error
    plt.plot(predict_y_array, color="red", alpha=0.8, label="predict")
    plt.plot(test_y, color="green", alpha=0.8, label="real")
    plt.ylabel("Money")
    plt.legend(loc="upper right")
    wandb.log({"LowPrice" : plt})
    wandb.summary["low price mse"] = mse(predict_y_array, test_y)

def main():
    r = requests.post("http://140.116.86.242:8081/stock/api/v1/buy", data={"uname":u_id, "pass":password, "scode": scode, "svol": str(vol), "sell_price":str(price)})
    print(r)


if __name__ == "__main__":
    wandb.init(project='Stock', entity="baron")
    all_data = ReadData(2330, 20150301, 20220322)
    training_data = all_data.get_training_data
    model = train(training_data, training_data.shape[1]-1)
    test_x, test_y = all_data.get_testing_data
    #model = load_model("save_model", feature_number=training_data.shape[1]-1)
    evaluation(model, test_x, test_y)
    wandb.finish()
    