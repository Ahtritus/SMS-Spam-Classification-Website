from flask import Blueprint, render_template, request, url_for
from .models import LSTM_CNN, LSTM_Glove, CNN, CNN_Glove
from dataclasses import dataclass

# creating an empty list that stores the queries and their results. It is rendered every time
result_list = []
views = Blueprint('views', __name__)

# creating a dataclass to store the results of each query


@dataclass
class Results:
    msg: str
    label: str
    model: str

# routing to the home page, renders on starting the app


@views.route('/')
def main():
    return render_template("home.html")

# routing to the home page, renders on submitting a message to be checked


@views.route('/', methods=['POST'])
def home():

    # text stores the input message
    text = request.form['msg']

    result = ""

    # model stores the selected prediction model
    model = request.form['model']

    # the selected prediction model is called from models.py file and the result returned by the model is stored in the result variable
    if model == "CNN":
        result = CNN(text)
    elif model == "CNN_Glove":
        result = CNN_Glove(text)
    elif model == "LSTM_CNN":
        result = LSTM_CNN(text)
    elif model == "LSTM_Glove":
        result = LSTM_Glove(text)

    # object of the daaclass, stores the input message, result and the selected prediction model
    obj = Results(text, result, model)

    # the list is an arrays of the objects created above, each object is inserted to the beginning of the list
    result_list.insert(0, obj)

    length = len(result_list)

    # the home page is rendered and the result list and its length are passed as attributes to be shown on the screen
    return render_template('home.html', result_list=result_list, length=length)
