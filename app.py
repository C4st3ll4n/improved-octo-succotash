import pickle

import pandas as pd
from flask import Flask, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    data = data["data"]
    data_df = pd.DataFrame(data=data)
    with open("models/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    prediction = model.predict(data_df).tolist()
    return prediction


if __name__ == '__main__':
    app.run()
