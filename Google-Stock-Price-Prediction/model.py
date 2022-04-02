import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM                                      # IMPORTING LIBRARIES
from keras.layers import Dropout

train_dataset  = pd.read_csv("./trainset.csv")                     # LOADING TRAIN DATASET
trainset = train_dataset.iloc[:,1:2].values

sc = MinMaxScaler(feature_range = (0,1))
scaled_trainset = sc.fit_transform(trainset)

X = []
Y = []

for i in range(60,1259):
    X.append(scaled_trainset[i-60:i, 0])                           # PREPROCESSING THE DATASET
    Y.append(scaled_trainset[i,0])

X,Y = np.array(X),np.array(Y)
X = np.reshape(X, (X.shape[0],X.shape[1],1))

epochs = 100
batch_size = 32                                                    # INITIALIZING NUMBER OF EPOCHS AND BATCH SIZE

def train():
 
    model = Sequential()                                                                  
    model.add(LSTM(units = 50,return_sequences = True,input_shape = (X.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50,return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50,return_sequences = True))                                   # INITIALIZING AND TRAINING THE MODEL
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam',loss = 'mean_squared_error')
    model.fit(X,Y,epochs = epochs, batch_size = batch_size)

    return model

def predict(model):

    test_dataset =pd.read_csv("./testset.csv")                                            # LOADING THE TEST DATASET
    true_stock_price = test_dataset.iloc[:,1:2].values
    full_dataset = pd.concat((train_dataset['Open'],test_dataset['Open']),axis = 0)

    inputs = full_dataset[len(full_dataset) - len(test_dataset)-60:].values               
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60,185):
        X_test.append(inputs[i-60:i,0])                                                    

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    predicted_stock_price = model.predict(X_test)                                         # PREDICTING THE STOCK PRICE
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    return true_stock_price,predicted_stock_price

def plot(true_stock_price,predicted_stock_price):

    plt.plot(true_stock_price,color = 'green', label = 'REAL PRICE')                      # PLOTTING THE TRUE AND PREDICTED STOCK PRICE
    plt.plot(predicted_stock_price, color = 'orange', label = 'PREDICTED PRICE')
    plt.title('GOOGLE STOCK PRICE PREDICTION')
    plt.xlabel('TIME')
    plt.ylabel('GOOGLE STOCK PRICE')
    plt.legend()
    plt.show()

def main():

    model = train ()
    true_stock_price , predicted_stock_price = predict (model)
    plot (true_stock_price,predicted_stock_price)

if __name__ == "__main__":
    main ()