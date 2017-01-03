import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
 

def main():
    development_data=pd.read_csv('DevData.csv')
    print("The total number of data:", development_data.size)
    #Clean all the columns with NA
    cleaned = development_data.dropna()
    print("The total number of data(After drop NAN):", cleaned.size)
    faults=np.array(cleaned['Faults'])
    print("type of faults", type(faults))
    FP=np.array(cleaned['Actual FP'])
    Schedule=np.array(cleaned['Actual Schedule'])
    SLOC=np.array(cleaned['Actual SLOC'])
    Req_review=cleaned['Req. Review']
    Design_review=cleaned['Design Review']
    Test_review=cleaned['Test Review']
    Code_review=np.array(cleaned['Code Review'])
    UT_defects=cleaned['UT Defects']
    fault_density=faults/FP
    SLOC_per_FP=SLOC/FP
    print(Code_review)  

    X=[]
    for i in np.arange(len(FP)):
        X.append([SLOC_per_FP[i], Schedule[i]])
    print(type(X))
  

    X_scaled = preprocessing.scale(X)
    print(X_scaled)
    y_pred=KMeans(n_clusters=2).fit_predict(X)
    A=np.array(X)

    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.scatter(A[:, 0], A[:, 1], c=y_pred)
    plt.title("One Point out of Control(no scaled)")

    y_pred=KMeans(n_clusters=2).fit_predict(X_scaled)
    A=np.array(X_scaled)
    plt.subplot(222)
    plt.scatter(A[:, 0], A[:, 1], c=y_pred)
    plt.title("One Point out of Control")

    #Remove the Point from the array
    for i in np.arange(len(FP)):
        if y_pred[i]==1:
            X.pop(i)
            print(i)
      

    X_scaled = preprocessing.scale(X)    
    y_pred=KMeans(n_clusters=2).fit_predict(X_scaled)
    A=np.array(X_scaled)       
    plt.subplot(223)
    plt.scatter(A[:, 0], A[:, 1], c=y_pred)
    plt.title("2 Clusters")

    y_pred=KMeans(n_clusters=4).fit_predict(X_scaled)
    A=np.array(X_scaled)       
    plt.subplot(224)
    plt.scatter(A[:, 0], A[:, 1], c=y_pred)
    plt.title("3 Clusters")
    plt.show()

    y_pred=pd.DataFrame(y_pred)
    y_pred.to_csv('y_pred.csv')

if __name__ == '__main__':
    main() 


