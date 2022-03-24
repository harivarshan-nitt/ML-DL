import numpy as np
import pandas as pd
import matplotlib.pyplot as plt                                                        # Importing libraries

dataset = pd.read_csv("./house_price.csv")                                                    # Reading dataset
dataset.drop_duplicates(subset='sqft', keep=False, inplace=True)

sqft = np.array(dataset['sqft'])
price = np.array(dataset['price'])
s = np.size(sqft)
costs = []                                                                             # Declaring cost list
iterations=200

def train(learning_rate=0.000000001):                                                  # train function
    m = 0
    c = 0
    price_predicted = np.zeros(s)
    for i in range( iterations ):
        price_predicted=m*sqft + c                                                     # Predicted Price
        costs.append((1 /(2*s))*(np.sum(price_predicted - price)**2))                  # Computing cost
        m = m - (learning_rate * ((1/s) * np.sum((price_predicted - price)*sqft)))     # Gradient descent
        c = c - (learning_rate * ((1/s) * np.sum((price_predicted - price))))
    return m,c

def predict(m,c):                                                                      # Prediction using m,c
    sqft_input = float(input("Enter Space in Sqft to predict price "))
    print("The predicted price for "+str(sqft_input)+" is $"+ str(m*sqft_input+c))

def plot(m,c):                                                                         # Plotting fitting curve
    plt.subplot(1, 2, 1)
    plt.scatter(sqft,price, color= 'red')
    plt.title ("Best Fit Curve")
    plt.xlabel("Sq ft")
    plt.ylabel("Price in $")
    plt.plot(sqft, m*sqft+c)

    plt.subplot(1, 2, 2)                                                               # Plotting cost function
    plt.title ("Cost Function")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.plot(range(iterations), costs)    

    plt.show()

def main():
    m,c = train()
    plot(m,c)
    predict(m,c)

if __name__ == "__main__":
    main()


