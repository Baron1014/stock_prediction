import requests 
import argparse
import json
import torch
import wandb
import logger
import numpy as np
from datetime import datetime
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

# Model para
learning_rate = 0.0001
EPOCH = 100000

logger = logger.create_logger('predict_low_price', 'log/predict_low_price_log.log')

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

class ReadInference:
    def __init__(self, s_date, e_date):
        api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(2330, s_date, e_date)
        r = requests.get(api_url)
        history_info = json.loads(r.text)['data']
        # data rerset low_price to next day
        history_data = [
            [data['capacity'], data['turnover'], data["open"], data["high"],data["close"],data["change"],data["transaction_volume"]] 
            for data in history_info
        ]
        history_data = np.array(history_data)

        train_data = ReadData()
        nor_x = train_data.normalize(history_data)
        self.__inference_data = nor_x[0, :].reshape(1, -1)

    @property
    def get_inference_data(self):
        return  torch.Tensor(self.__inference_data)

class ReadData():
    def __init__(self, stock=2330, s_date=20150301, e_date=20220322):
        api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock, s_date, e_date)
        r = requests.get(api_url)
        history_info = json.loads(r.text)['data']
        # data rerset low_price to next day
        target = [data["low"] for i, data in enumerate(history_info) if i != len(history_info)-1]
        history_data = [
            [data['capacity'], data['turnover'], data["open"], data["high"],data["close"],data["change"],data["transaction_volume"], target[i]] 
            for i, data in enumerate(history_info) if i != len(history_info)-1
        ]
        # date: new -> old, change to old -> new
        history_data.reverse()
        history_data = np.array(history_data)
        
        self.__training_data, self.__testing_data = train_test_split(history_data, test_size=0.1, shuffle=False)
        self.__mean = np.mean(self.__training_data[:, :-1], axis=0)
        self.__std = np.std(self.__training_data[:, :-1], axis=0)
        self.__all_data = history_data
        
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
    
    @property
    def get_all_data(self):
        # normalize x: input type "numpy array"
        y = self.__all_data[:, -1].reshape(-1, 1)
        nor_x = self.normalize(self.__all_data[:, :-1])

        return torch.Tensor(nor_x), y


    def normalize(self, data):
        return  (data - self.__mean) / self.__std

def train(training_data, feature_number):
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

def predict():
    today = datetime.now().strftime('%Y%m%d')
    all_data = ReadInference(20220301, int(today))
    predict_data = all_data.get_inference_data
    model = load_model("save_model", feature_number=predict_data.shape[1])
    predict_value = model(predict_data.reshape(1, *predict_data.shape))

    return "%.1f" % predict_value.item()


def load_model(model_name, feature_number):
    device = torch.device('cpu')
    model = RNN(feature_number)
    model.load_state_dict(torch.load("models/{}.pth".format(model_name), map_location=device))

    return model

def evaluation(model, test_x, test_y, history_entity):
    predict_y = model(test_x.reshape(1, *test_x.shape))
    predict_y_array = predict_y.cpu().detach().numpy().reshape(-1, 1)
    print(predict_y_array.shape)

    # low price error
    plt.plot(predict_y_array, color="red", alpha=0.8, label="predict")
    plt.plot(test_y, color="green", alpha=0.8, label="real")
    plt.ylabel("Money")
    plt.legend(loc="upper right")
    wandb.log({"LowPrice" : plt})
    # line_data = np.concatenate((predict_y_array, test_y), axis=1)
    # idx = np.array([float(i) for i in range(line_data.shape[0])])
    #line_data = np.concatenate((idx, line_data), axis=1)
    #table = wandb.Table(data=line_data, columns=["idx", "predict", "real"])
    # wandb.log({"predict low price":  wandb.plot.line_series(xs=idx, ys=line_data.reshape(2, -1), keys=["predict", "real"], title="Predict & Real LowPrice")})
    #wandb.log({"real low price":  wandb.plot.line(table, "idx", "real", title="Real LowPrice")})
    wandb.summary["low price mse"] = mse(predict_y_array, test_y)

    # plot history curve
    his_x, his_y = history_entity.get_all_data
    his_y_hat = model(his_x.reshape(1, *his_x.shape))
    his_y_hat = his_y_hat.cpu().detach().numpy().reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.plot(his_y_hat, color="red", alpha=0.8, label="predict")
    ax.plot(his_y, color="green", alpha=0.8, label="real")
    ax.legend(loc="upper right")
    wandb.log({"History Low price" : ax})


def main():
    low_price = predict()
    r = requests.post("http://140.116.86.242:8081/stock/api/v1/buy", data={"uname":u_id, "pass":password, "scode": scode, "svol": str(vol), "sell_price":str(low_price)})
    print(r)
    print(low_price)
    logger.info("@Low price:{} @Low number:{}".format(low_price, vol))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--predict',
                        action='store_true',
                        help='Execute prediction.')
    
    parser.add_argument('--train',
                        action='store_true',
                        help='Execute training.')
    
    args = parser.parse_args()
    if args.predict:
        week_day = datetime.today().strftime('%A')
        if week_day not in ["Saturday", "Sunday"]:
            print("Today is {}".format(week_day))
            main()
    elif args.train:
        wandb.init(project='Stock', entity="baron")
        all_data = ReadData()
        training_data = all_data.get_training_data
        model = train(training_data, training_data.shape[1]-1)
        test_x, test_y = all_data.get_testing_data
        evaluation(model, test_x, test_y, history_entity = all_data)
        wandb.finish()
    else:
        wandb.init(project='Stock', entity="baron")
        all_data = ReadData()
        training_data = all_data.get_training_data
        test_x, test_y = all_data.get_testing_data
        model = load_model("save_model", feature_number=training_data.shape[1]-1)
        evaluation(model, test_x, test_y, history_entity = all_data)
        wandb.finish()

    