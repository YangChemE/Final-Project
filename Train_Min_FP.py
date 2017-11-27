
from random import *


def readDATA(N):
    
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    ##read the data 
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", 
                        names = ["class","cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing",
                                "gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring",
                                "stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type",
                                "veil-color","ring-number","ring-type","spore-print-color","population","habitat"],
                        header = None)


    ##remove the feature with missing values
    df = df.drop('stalk-root',1)



    feature_columns = df.columns[1:]

    #Check chi2 significance of 22 features    
    from sklearn.feature_selection import chi2, SelectKBest
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    numeric_data = pd.DataFrame()
    for f in feature_columns:
        numeric_data[f] = le.fit_transform(df[f])
    
    chi_statics, p_values = chi2(numeric_data, df['class'])

    chi2_result = pd.DataFrame({'features': feature_columns, 'chi2_statics': chi_statics, 'p_values': p_values})
    chi2_result.dropna(axis=0, how='any', inplace=True)

    
    #choose most descriptive features for learning
    top_features = chi2_result.sort_values(by='chi2_statics', ascending=False)['features'].head(N).values

    
    feature_important = pd.DataFrame()
    for j in top_features:
        dumm = df[j].str.get_dummies()
        dumm.columns = ['{}_{}'.format(j, v) for v in dumm.columns]
        feature_important = pd.concat([feature_important, dumm], axis=1)
    feature_important['class'] = df['class']
    row, column = feature_important.shape

    #1-hot encoding features
    enc = LabelEncoder()
    enc.fit(feature_important[feature_important.columns[column-1]])
    feature_important[feature_important.columns[column-1]] = enc.transform(feature_important[feature_important.columns[column-1]])

    print(feature_important.shape)
    #train and test split.
    train, test = train_test_split(feature_important,train_size = 0.8)    

    X_train = train.drop('class',1)
    X_test = test.drop('class',1)
    y_train = train['class']
    y_test = test['class']
    return X_train, X_test, y_train, y_test
    

def LogReg(X_train, y_train, X_test, y_test):
    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    a=clf.score(X_train, y_train)
    b=clf.score(X_test, y_test)
    

    #confusion matrix
    from sklearn.metrics import confusion_matrix
    p = clf.predict(X_test)
    cm = confusion_matrix(y_test,p)
    #print(cm)
    tn, fp, fn, tp = cm.ravel()
    Precision = (tp/(tp+fp))
    Recall = (tp/(tp+fn))
    return a,b,fp
    

def KNN(X_train, y_train, X_test, y_test):
    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit (X_train, y_train)
    p = knn.predict(X_test)
    a = knn.score(X_train, y_train)
    b = knn.score(X_test, y_test)
    


    #confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,p)
    #print(cm)
    tn, fp, fn, tp = cm.ravel()
    Precision = (tp/(tp+fp))
    Recall = (tp/(tp+fn))
    return a,b,fp

def NN(X_train, y_train, X_test, y_test):
    # Neural Network
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes = (30, 30), activation='logistic', alpha=0.001, solver='lbfgs', learning_rate='constant')
    mlp.fit(X_train, y_train)
    p = mlp.predict(X_test)
    a = mlp.score(X_train, y_train)
    b = mlp.score(X_test, y_test)




    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,p)
    #print(cm)
    tn, fp, fn, tp = cm.ravel()
    Precision = (tp/(tp+fp))
    Recall = (tp/(tp+fn))
    return a,b,fp
    

def Tree(X_train, y_train, X_test, y_test):    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree as TREE
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    p = tree.predict(X_test)
    a = tree.score(X_train, y_train)
    b = tree.score(X_test, y_test)
    
    
    from sklearn.metrics import confusion_matrix
    #confusion matrix
    cm = confusion_matrix(y_test,p)
    #print(cm)
    tn, fp, fn, tp = cm.ravel()
    Precision = (tp/(tp+fp))
    Recall = (tp/(tp+fn))
    
    
    import graphviz 
    dot_data = TREE.export_graphviz(tree, out_file=None,                          
                         filled=True, rounded=True,  
                         special_characters=True)  
    %matplotlib inline
    graph = graphviz.Source(dot_data)  
    graph
    return a,b,fp

    
def NB(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()

    nb.fit(X_train, y_train)

    p = nb.predict(X_test)
    a = nb.score(X_train, y_train)
    b = nb.score(X_test, y_test)
    
    


    #confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,p)
    #print(cm)
    tn, fp, fn, tp = cm.ravel()
    Precision = (tp/(tp+fp))
    Recall = (tp/(tp+fn))
    return a,b,fp

Accu_Train_LogReg = 0
Accu_Test_LogReg = 0
FP_LogReg = 0

Accu_Train_KNN = 0
Accu_Test_KNN = 0
FP_KNN = 0

Accu_Train_NN = 0
Accu_Test_NN = 0
FP_NN = 0

Accu_Train_Tree = 0
Accu_Test_Tree = 0
FP_Tree = 0

Accu_Train_NB = 0
Accu_Test_NB = 0
FP_NB = 0

for i in range (100):

    #Read and split data
    X_train, X_test, y_train, y_test = readDATA(6) #given the number of feature will be used in training

    #logistic regression
    a,b,c = LogReg(X_train, y_train, X_test, y_test)
    Accu_Train_LogReg, Accu_Test_LogReg, FP_LogReg = Accu_Train_LogReg+a, Accu_Test_LogReg+b, FP_LogReg+c
    
    #KNN
    a,b,c = KNN(X_train, y_train, X_test, y_test)
    Accu_Train_KNN, Accu_Test_KNN, FP_KNN = Accu_Train_KNN+a, Accu_Test_KNN+b, FP_KNN+c
    
    #Neural Network
    a,b,c = NN(X_train, y_train, X_test, y_test)
    Accu_Train_NN, Accu_Test_NN, FP_NN = Accu_Train_NN+a, Accu_Test_NN+b, FP_NN+c
    
    #Decision Tree
    a,b,c = Tree(X_train, y_train, X_test, y_test)
    Accu_Train_Tree, Accu_Test_Tree, FP_Tree = Accu_Train_Tree+a, Accu_Test_Tree+b, FP_Tree+c    
    
    #Naive Bayes
    a,b,c = NB(X_train, y_train, X_test, y_test)
    Accu_Train_NB, Accu_Test_NB, FP_NB = Accu_Train_NB+a, Accu_Test_NB+b, FP_NB+c
    
    i+1
    print (i, 'th training')
    
Accu_Train_LogReg_Avg = Accu_Train_LogReg/100 
Accu_Test_LogReg_Avg = Accu_Train_LogReg/100
FP_LogReg_Avg = FP_LogReg/100

Accu_Train_KNN_Avg = Accu_Train_KNN/100 
Accu_Test_KNN_Avg = Accu_Train_KNN/100
FP_KNN_Avg = FP_KNN/100

Accu_Train_NN_Avg = Accu_Train_NN/100 
Accu_Test_NN_Avg = Accu_Train_NN/100
FP_NN_Avg = FP_NN/100

Accu_Train_Tree_Avg = Accu_Train_Tree/100 
Accu_Test_Tree_Avg = Accu_Train_Tree/100
FP_Tree_Avg = FP_Tree/100

Accu_Train_NB_Avg = Accu_Train_NB/100 
Accu_Test_NB_Avg = Accu_Train_NB/100
FP_NB_Avg = FP_NB/100



print('Avg Accuracy Train LogReg: ', Accu_Train_LogReg_Avg)
print('Avg Accuracy Train KNN: ', Accu_Train_KNN_Avg)
print('Avg Accuracy Train NN: ', Accu_Train_NN_Avg)
print('Avg Accuracy Train Tree: ', Accu_Train_Tree_Avg)
print('Avg Accuracy Train NB: ', Accu_Train_NB_Avg)

print('Avg Accuracy Test LogReg: ', Accu_Test_LogReg_Avg)
print('Avg Accuracy Test KNN: ', Accu_Test_KNN_Avg)
print('Avg Accuracy Test NN: ', Accu_Test_NN_Avg)
print('Avg Accuracy Test Tree: ', Accu_Test_Tree_Avg)
print('Avg Accuracy Test NB: ', Accu_Test_NB_Avg)

print('Avg False Positive LogReg: ', FP_LogReg_Avg)
print('Avg False Positive KNN: ', FP_KNN_Avg)
print('Avg False Positive NN: ', FP_NN_Avg)
print('Avg False Positive Tree: ', FP_Tree_Avg)
print('Avg False Positive NB: ', FP_NB_Avg)
graph
