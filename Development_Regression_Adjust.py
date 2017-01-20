import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def data_list(Numpyarray, X):
    for i in range(0, len(Numpyarray)):
        X.append([Numpyarray[i]])

def main():
    development_data=pd.read_csv('DevData_adjust.csv')
    print("The total number of data:", development_data.size)
    fig=plt.figure(figsize=(50,50))
    
    #cleaned.shape[1]
    cleaned = np.array(development_data.dropna())

    #step1: find the relationship between factors
    for i in range(0, cleaned.shape[1]):
        for j in range(0, cleaned.shape[1]):
            X=[]
            Y=[]
            data_list(cleaned[:, i], X)
            data_list(cleaned[:, j], Y)
            ax=fig.add_subplot(12, 12, i*12+j+1)
            if i!=j:
                plt.xlabel(development_data.columns[i])
                plt.ylabel(development_data.columns[j])
                ax.scatter(X,Y)
                
            #else:
             #   ax.annotate("Series"+str(i), xycoords='axes fraction', ha="center", va="center")

    plt.savefig('relationship.pdf', format='pdf')

    #step2: find the outlier of the factors
    cleaned_scaled=preprocessing.scale(cleaned) #adjust the data to the same scale
    plt.clf()
    plt.figure()
    plt.title('outlier')
    plt.boxplot(cleaned_scaled)
    plt.xticks(range(0, len(development_data.columns)), development_data.columns, rotation=90)
    plt.savefig('outlier.pdf', format='pdf')
    
    #****************Model 1: Estimation model (SLOC/FP)******************
    #step 3: seperate the data into factors and response
    FP=cleaned[:, 1]
    LOC=cleaned[:, 2]
    
    #step 4: spreate the data into trian set and test set
    X_train_estimation, X_test_estimation, y_train_estimation, y_test_estimation=train_test_split(FP, LOC, random_state = 0)
    X_estimation_train = []
    y_estimation_train = []
    X_estimation_test = []
    y_estimation_test = []
    data_list(X_train_estimation, X_estimation_train)
    data_list(y_train_estimation, y_estimation_train)
    data_list(X_test_estimation, X_estimation_test)
    data_list(y_test_estimation, y_estimation_test)

    #step 5: Build the Regression model--LiearRegression, Rigid model, etc.
    #step 5.1: Build linear_regression Model/Chart
    model_1 = LinearRegression()
    model_1.fit(X_estimation_train, y_estimation_train)
    print("Model 1: the linear_regression score for train %f" %model_1.score(X_estimation_train, y_estimation_train))
    print("Model 1: the linear_regression score for test %f" %model_1.score(X_estimation_test, y_estimation_test))
    plt.clf()
    plt.figure()
    plt.xlabel('FP')
    plt.ylabel('SLOC')
    plt.scatter(X_estimation_train, y_estimation_train)
    line=np.linspace(0, 2500, 100)
    line1=[]
    data_list(line, line1)
    plt.plot(line1, model_1.predict(line1))
    

    #step 5.2: Build linear_regression model for 2 poly
    poly2=PolynomialFeatures(2)
    X_estimation_train2=poly2.fit_transform(X_estimation_train)
    X_estimation_test2=poly2.fit_transform(X_estimation_test)
    model_2 = LinearRegression()
    model_2.fit(X_estimation_train2, y_estimation_train)
    print("Model 1_2: the linear_regression score for train %f" %model_2.score(X_estimation_train2, y_estimation_train))
    print("Model 1_2: the linear_regression score for test %f" %model_2.score(X_estimation_test2, y_estimation_test))
    line2=poly2.fit_transform(line1)
    plt.plot(line1, model_2.predict(line2))
    plt.legend(['1-Dimension', '2-Dimension'])
    plt.savefig('Linearregression.pdf', format='pdf')
    
    #step 5.3: Build Ridig model
    model_Ridge1 = Ridge()
    model_Ridge1.fit(X_estimation_train, y_estimation_train)
    print("Rigid 1: the Ridge score for train %f" %model_Ridge1.score(X_estimation_train, y_estimation_train))
    print("Rigid 1: the linear_regression score for test %f" %model_Ridge1.score(X_estimation_test, y_estimation_test))

    #step 5.4: Build Ridig model for 2 poly
    model_Ridge2 = Ridge()
    model_Ridge2.fit(X_estimation_train2, y_estimation_train)
    print("Rigid 2: the Ridge score for train %f" %model_Ridge2.score(X_estimation_train2, y_estimation_train))
    print("Rigid 2: the linear_regression score for test %f" %model_Ridge2.score(X_estimation_test2, y_estimation_test))
    
    #step 5.5: Adjust the Alpha to see the result/coefs
    model_Ridge1 = Ridge(alpha=1000)
    model_Ridge1.fit(X_estimation_train, y_estimation_train)
    print("Rigid 1: the Ridge score for train %f" %model_Ridge1.score(X_estimation_train, y_estimation_train))
    print("Rigid 1: the linear_regression score for test %f" %model_Ridge1.score(X_estimation_test, y_estimation_test))

    
    #*****************Model 2: Effort & Size Economic Model***********************
    #Model 3: Language judge Model
    
    
    #Model 4: Fault Density Model
    
    
    
if __name__ == '__main__':
    main() 


