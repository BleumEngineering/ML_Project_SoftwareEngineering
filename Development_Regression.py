import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

def data_list(Numpyarray, X):
    for i in range(0, len(Numpyarray)):
        X.append([Numpyarray[i]])

def main():
    development_data=pd.read_csv('DevData.csv')
    print("The total number of data:", development_data.size)
    fig=plt.figure(figsize=(10,10))
    
    #cleaned.shape[1]
    cleaned = np.array(development_data.dropna())
    for i in range(0, 5):
        for j in range(0, 5):
            print(i, j, '\n')
            X=[]
            Y=[]
            data_list(cleaned[:, i], X)
            data_list(cleaned[:, j], Y)
            ax=fig.add_subplot(12, 12, i*12+j+1)
            ax.scatter(X,Y)

    plt.savefig('relationship.pdf', format='pdf')


if __name__ == '__main__':
    main() 


