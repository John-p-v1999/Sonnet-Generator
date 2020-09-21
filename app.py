import flask
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sample(preds,temperature=1.0):
  preds =np.asarray(preds).astype('float64')
  preds=np.log(preds)/temperature
  exp_preds=np.exp(preds)
  preds=exp_preds/np.sum(exp_preds)
  return np.argmax(preds)
app= Flask(__name__)
@app.route('/')
def man():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def home():
    model = keras.Sequential()
    model.add(keras.layers.Embedding(20596, 100, input_length=159))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(128)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dense(20596, activation='softmax'))
    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.load_weights('trained_weights5') # loading trained model weights
    str_final=request.form['a']
    temperature=request.form['b']
    tokenizer=pickle.load(open('tokenizer.pickle','rb'))
    inp = tokenizer.texts_to_sequences([str_final])
    padded_test = np.array(pad_sequences(inp, maxlen=159))
    index = tokenizer.word_index
    index = {value: key for key, value in index.items()}
    temperature=float(temperature)
    for i in range(98):

        p_prob = model.predict(padded_test)
        pred = sample(p_prob, temperature)
        str_pred = index[pred]
        if str_pred == 'eos':
            str_final = str_final + " "
            inp = tokenizer.texts_to_sequences([str_final + 'eos'])
        else:
            str_final = str_final + " " + str_pred
            inp = tokenizer.texts_to_sequences([str_final])
        padded_test = np.array(pad_sequences(inp, maxlen=159))
        if ((i+1) % 7 == 0):
            str_final = str_final + '\n'
    return render_template('result.html', data = str_final)
if __name__ == '__main__':
    app.run(debug=True)
