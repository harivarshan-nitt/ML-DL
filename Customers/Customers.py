import numpy as np
import pandas as pd
import matplotlib.pyplot as plt                                       # Importing libraries

dataset = pd.read_csv("./Customers.csv")                              # Reading dataset
income = np.array(dataset['Income'])
spending_score = np.array(dataset['Spending_Score'])

Data = []
Groups = []                                                           # Group list
centroids = []                                                        # Centroid list

iterations = 100                                                       
K = 5                                                               

class Point:                                                          # Class definition for Point object
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def distance(self,Y):                                             # Distance function 
        return ((self.x - Y.x)**2 + (self.y - Y.y)**2)**0.5

def init_dataset():                                                   # Initializing Dataset as list of Point Objects
    for i in range (np.size(income)):
        Data.append(Point(income[i],spending_score[i]))

def init_centroids():                                                 # Initializing Centroids
    min_income = np.min(income)
    max_income = np.max(income)
    min_spending_score = np.min(spending_score)
    max_spending_score = np.max(spending_score)
    centroids.append(Point(min_income,min_spending_score))
    centroids.append(Point(max_income,min_spending_score))
    centroids.append(Point(max_income,max_spending_score))
    centroids.append(Point(min_income,max_spending_score))
    centroids.append(Point((max_income-min_income)/2,(max_spending_score-min_spending_score)/2))

def init_groups():                                                    # Initializing empty groups
    for i in range(K):
        G = []
        Groups.append(G)

def Group():                                                          # Function for grouping Points based on shortest distance from centroids
    for i in range(K):
        Groups[i] = []
    for D in Data:
        I = 0
        C = centroids[0]
        for i in range(K):
            if D.distance(centroids[i]) < D.distance(C):
                C = centroids[i]
                I = i
        Groups[I].append(D)

def update_centroid():                                               # Function for updating centroids by taking average
    for i in range(K):
        X = 0
        Y = 0
        for G in Groups[i]:
            X = X + G.x
            Y = Y + G.y
        if len(Groups[i])!=0:
            centroids[i] = Point(X/len(Groups[i]),Y/len(Groups[i]))

def convert_to_array(W):                                             # Function which converts Point object array to list
    X = []
    Y = []
    for w in W:
        X.append(w.x)
        Y.append(w.y)
    return X,Y

def plot():                                                          # Function which plots clustered groups
    for G in Groups:
        X ,Y = convert_to_array(G)
        plt.scatter(X,Y)
    X,Y = convert_to_array(centroids)
    plt.scatter(X,Y, color= 'black')
    plt.title ("K Means Clustering of Customers")
    plt.xlabel("Income in 1000$")
    plt.ylabel("Spending_Score (1-10)")
    plt.show()

def main():
    init_dataset()                                                   # Initializing
    init_groups()
    init_centroids()
    Group()
    for k in range(iterations):                                      # Iterating
        update_centroid()
        Group()
    plot()

if __name__ == "__main__":
    main()