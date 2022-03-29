import os
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

# Model para
learning_rate = 0.0001
EPOCH = 10000

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
        self.input = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LeakyReLU()
        )
        self.rnn = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.input(x)
        r_out, (h_n, h_c) = self.rnn(h, None)  # None 表示 hidden state 會用全0的 state
        out = self.out(r_out)
        return out

class ReadInference:
    def __init__(self, s_date, e_date):
        api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(2330, s_date, e_date)
        r = requests.get(api_url)
        history_info = json.loads(r.text)['data']
        # data rerset low_price to next day
        self.__target = [data["low"] for i, data in enumerate(history_info) if i != len(history_info)-1]
        self.__his_date = [datetime.fromtimestamp(data["date"]) for i, data in enumerate(history_info) if i != len(history_info)-1]
        history_data = [
            [data['capacity'], data['turnover'], data["open"], data["high"],data["close"],data["change"],data["transaction_volume"]] 
            for i, data in enumerate(history_info)
        ]
        # date: new -> old, change to old -> new
        history_data.reverse()
        self.__his_date.reverse()
        self.__target.reverse()
        history_data = np.array(history_data)

        train_data = ReadData()
        self.__history_data_nor = train_data.normalize(history_data)
        

    @property
    def get_evaluation_data(self):
        return  torch.Tensor(self.__history_data_nor), self.__target, self.__his_date

class ReadData():
    def __init__(self, stock=2330, s_date=20150301, e_date=20220322):
        api_url = "http://140.116.86.242:8081/stock/api/v1/api_get_stock_info_from_date_json/{}/{}/{}".format(stock, s_date, e_date)
        r = requests.get(api_url)
        history_info = json.loads(r.text)['data']
        # data rerset low_price to next day
        target = [data["low"] for i, data in enumerate(history_info) if i != len(history_info)-1]
        his_date = [datetime.fromtimestamp(data["date"]) for i, data in enumerate(history_info) if i != len(history_info)-1]
        history_data = [
            [data['capacity'], data['turnover'], data["open"], data["high"],data["close"],data["change"],data["transaction_volume"], target[i-1]] 
            for i, data in enumerate(history_info) if i != 0
        ]
        # date: new -> old, change to old -> new
        history_data.reverse()
        his_date.reverse()
        history_data = np.array(history_data)
        
        self.__training_data, self.__testing_data, self.__training_date, self.__testing_date = train_test_split(history_data, his_date, test_size=0.1, shuffle=False)
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

    @property
    def get_datetime(self):
        return self.__training_date, self.__testing_date

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

    #torch.save(model.state_dict(), "./models/save_model.pth")
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'final_model.pth'))
    torch.onnx.export(model, tx, "model_final.onnx")
    wandb.save("model_final.onnx")
    return model

def predict(model_id):
    today = datetime.now().strftime('%Y%m%d')
    all_data = ReadInference(20210810, int(today))

    # plot test result
    test_data, test_y, test_date = all_data.get_evaluation_data
    model = load_model(model_id, feature_number=test_data.shape[1])
    eval_hat = model(test_data.reshape(1, *test_data.shape))
    eval_hat = eval_hat.cpu().detach().numpy().reshape(-1, 1)
    predict_value = eval_hat[-1]
    plot_testing_result(test_date, eval_hat[:-1, :], test_y)

    return "%.1f" % predict_value.item()


def load_model(run_id, feature_number):
    api = wandb.Api()
    run = api.run("baron/Stock/{}".format(run_id))
    run.file("final_model.pth").download("./models/", replace=True)

    device = torch.device('cpu')
    model = RNN(feature_number)
    #model.load_state_dict(torch.load("models/{}.pth".format(model_name), map_location=device))
    model.load_state_dict(torch.load("./models/final_model.pth", map_location=device))

    return model

def evaluation(model, test_x, test_y, history_entity):
    train_date, test_date = history_entity.get_datetime
    predict_y = model(test_x.reshape(1, *test_x.shape))
    predict_y_array = predict_y.cpu().detach().numpy().reshape(-1, 1)
    print(predict_y_array.shape)
    # low price error
    plot_testing_result(test_date, predict_y_array, test_y)

    # plot history curve
    his_x, his_y = history_entity.get_all_data
    his_y_hat = model(his_x.reshape(1, *his_x.shape))
    his_y_hat = his_y_hat.cpu().detach().numpy().reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.plot(train_date+test_date, his_y_hat, color="red", alpha=0.8, label="predict")
    ax.plot(train_date+test_date,his_y, color="green", alpha=0.8, label="real")
    ax.legend(loc="upper right")
    wandb.log({"History Low price" : ax})

def plot_testing_result(test_date, y_hat, real_y):
    # low price error
    plt.plot(test_date, y_hat, color="red", alpha=0.8, label="predict")
    plt.plot(test_date, real_y, color="green", alpha=0.8, label="real")
    plt.ylabel("Money")
    plt.legend(loc="upper right")
    wandb.log({"LowPrice" : plt})
    # line_data = np.concatenate((predict_y_array, test_y), axis=1)
    # idx = np.array([float(i) for i in range(line_data.shape[0])])
    #line_data = np.concatenate((idx, line_data), axis=1)
    #table = wandb.Table(data=line_data, columns=["idx", "predict", "real"])
    # wandb.log({"predict low price":  wandb.plot.line_series(xs=idx, ys=line_data.reshape(2, -1), keys=["predict", "real"], title="Predict & Real LowPrice")})
    #wandb.log({"real low price":  wandb.plot.line(table, "idx", "real", title="Real LowPrice")})
    wandb.summary["low price mse"] = mse(y_hat, real_y)

def main(model_id):
    # stock info
    u_id = "NM6101080"
    password = "as987654"
    scode = "2330"
    vol = 5

    low_price = predict(model_id)
    r = requests.post("http://140.116.86.242:8081/stock/api/v1/buy", data={"account":u_id, "password":password, "stock_code": scode, "stock_shares": str(vol), "stock_price":str(low_price)}).json()
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
    model_id = "t97eyk6r"
    if args.predict:
        week_day = datetime.today().strftime('%A')
        if week_day not in ["Saturday", "Sunday"]:
            wandb.init(project='Stock', entity="baron")
            print("Today is {}".format(week_day))
            main(model_id)
            wandb.finish()

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
        model = load_model(model_id, feature_number=training_data.shape[1]-1)
        evaluation(model, test_x, test_y, history_entity = all_data)
        wandb.finish()
