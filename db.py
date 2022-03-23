from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from numpy import insert

app = Flask(__name__)

# MySql datebase
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///./stock.db"
db = SQLAlchemy(app)

class Predict(db.Model):
    __tablename__ = 'predict_2330'
    pid = db.Column(db.Integer, primary_key=True)
    low_price_hat = db.Column(db.Float, nullable=False)
    high_price_hat = db.Column(db.Float, nullable=False)
    low_number = db.Column(db.Integer, nullable=False)
    high_number = db.Column(db.Integer, nullable=False)
    insert_time = db.Column(db.DateTime, default=datetime.now)

    def __init__(self, low_price, high_price, low_number, high_number):
        self.low_price_hat = low_price
        self.high_price_hat = high_price
        self.low_number = low_number
        self.high_number = high_number

def insert_predict_data(low_price, high_price, low_number, high_number):
    data = Predict(low_price, high_price, low_number, high_number)
    insert(data)

# insert data entity to database
def insert(db_entity):
    db.session.add(db_entity)
    db.session.commit()

if __name__=="__main__":
    db.create_all()