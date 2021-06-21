import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, tokenizer_from_json


# the input message is converted into a vector array by tokenizing it with the pre-trained tokenizer.json
def data_tokenize(msg):
    with open('./Models/tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    TestSMS = [msg]
    vectSMS = tokenizer.texts_to_sequences(np.array(TestSMS))
    vectSMS = pad_sequences(vectSMS, padding='post', maxlen=60)
    return vectSMS


# if the input prediction model is CNN-LSTM, this function is called
def LSTM_CNN(msg):

    # the input message is tokenized
    vectSMS = data_tokenize(msg)

    # the pre-trained LSTM_CNN model is loaded
    new_model = tf.keras.models.load_model(
        "./Models/lstm_cnn.h5", compile=False)

    # the prediction of the model is stored
    prediction = new_model.predict(vectSMS)

    # prediction[0] is Ham while prediction[1] is spam, this is due to one hot encoding implemented during the creation of the model
    # comparing probablilities of the message to be Ham or Spam
    if prediction[0][0] > prediction[0][1]:
        return "Ham"
    else:
        return "Spam"


# if the input prediction model is CNN-LSTM-GloVe, this function is called
def LSTM_Glove(msg):

    # the input message is tokenized
    vectSMS = data_tokenize(msg)

    # the pre-trained LSTM_CNN model is loaded
    new_model = tf.keras.models.load_model(
        "./Models/lstm_glove.h5", compile=False)

    # the prediction of the model is stored
    prediction = new_model.predict(vectSMS)

    # prediction[0] is Ham while prediction[1] is spam, this is due to one hot encoding implemented during the creation of the model
    # comparing probablilities of the message to be Ham or Spam
    if prediction[0][0] > prediction[0][1]:
        return "Ham"
    else:
        return "Spam"


# if the input prediction model is CNN, this function is called
def CNN(msg):

    # the input message is tokenized
    vectSMS = data_tokenize(msg)

    # the pre-trained LSTM_CNN model is loaded
    new_model = tf.keras.models.load_model("./Models/cnn.h5", compile=False)

    # the prediction of the model is stored
    prediction = new_model.predict(vectSMS)

    # prediction[0] is Ham while prediction[1] is spam, this is due to one hot encoding implemented during the creation of the model
    # comparing probablilities of the message to be Ham or Spam
    if prediction[0][0] > prediction[0][1]:
        return "Ham"
    else:
        return "Spam"


# if the input prediction model is CNN-GloVe, this function is called
def CNN_Glove(msg):

    # the input message is tokenized
    vectSMS = data_tokenize(msg)

    # the pre-trained LSTM_CNN model is loaded
    new_model = tf.keras.models.load_model(
        "./Models/cnn_glove.h5", compile=False)

    # the prediction of the model is stored
    prediction = new_model.predict(vectSMS)

    # prediction[0] is Ham while prediction[1] is spam, this is due to one hot encoding implemented during the creation of the model
    # comparing probablilities of the message to be Ham or Spam
    if prediction[0][0] > prediction[0][1]:
        return "Ham"
    else:
        return "Spam"
