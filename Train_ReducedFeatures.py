
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

    #df.describe()

    ##remove the feature with missing values
    df = df.drop('stalk-root',1)




    feature_columns = df.columns[1:]
    for i, f in zip(np.arange(1, len(feature_columns) + 1), feature_columns):
        print('feature {:d}:\t{}'.format(i, f))



    #Check chi2 significance of 22 features    
    from sklearn.feature_selection import chi2, SelectKBest
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    numeric_data = pd.DataFrame()
    for f in feature_columns:
        numeric_data[f] = le.fit_transform(df[f])
    
    chi_statics, p_values = chi2(numeric_data, df['class'])

    chi2_result = pd.DataFrame({'features': feature_columns, 'chi2_statics': chi_statics})
    chi2_result.dropna(axis=0, how='any', inplace=True)

    print(chi2_result.sort_values(by='chi2_statics', ascending=False)[['features', 'chi2_statics']].reset_index().drop('index', axis=1))
    %matplotlib inline
    _ = chi2_result.sort_values(by='chi2_statics', ascending=True).set_index('features')['chi2_statics'].plot(kind='barh', logx=True, rot=-2)    
    
    #choose most descriptive features for learning
    top_features = chi2_result.sort_values(by='chi2_statics', ascending=False)['features'].head(N).values

    print('top ' ,N, ' most useful features are:')
    for i in top_features:
        print(i)
    
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
    print("Default Logistic Regression: training accuracy: ",a," testing accuracy: ",b)

    #a roc plot
    y_prob = clf.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
    y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
    clf.score(X_test, y_pred)
    from sklearn.metrics import roc_curve, auc
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    roc_auc
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    #confusion matrix
    from sklearn.metrics import confusion_matrix
    p = clf.predict(X_test)
    cm = confusion_matrix(y_test,p)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    Precision = (tp/(tp+fp))
    Recall = (tp/(tp+fn))
    print("tn: ",tn,"fp: ",fp,"fn:",fn,"tp: ",tp)
    print("Precision: ",Precision, "Recall: ", Recall, "F1: ", ((2*Precision*Recall)/(Precision+Recall)))
    return a,b

def KNN(X_train, y_train, X_test, y_test):
    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit (X_train, y_train)
    p = knn.predict(X_test)
    a = knn.score(X_train, y_train)
    b = knn.score(X_test, y_test)
    print("KNN: training accuracy: ",a," testing accuracy: ",b)

    #roc plot
    y_prob = knn.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
    y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
    knn.score(X_test, y_pred)
    from sklearn.metrics import roc_curve, auc
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    roc_auc
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    #confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,p)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    Precision = (tp/(tp+fp))
    Recall = (tp/(tp+fn))
    print("tn: ",tn,"fp: ",fp,"fn:",fn,"tp: ",tp)
    print("Precision: ",Precision, "Recall: ", Recall, "F1: ", ((2*Precision*Recall)/(Precision+Recall)))
    return a,b
    
def NN(X_train, y_train, X_test, y_test):
    # Neural Network
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes = (30, 30), activation='logistic', alpha=0.001, solver='lbfgs', learning_rate='constant')
    mlp.fit(X_train, y_train)
    p = mlp.predict(X_test)
    a = mlp.score(X_train, y_train)
    b = mlp.score(X_test, y_test)
    print("Default NN: training accuracy: ",a," testing accuracy: ",b)



    #roc plot
    y_prob = mlp.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
    y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
    mlp.score(X_test, y_pred)
    from sklearn.metrics import roc_curve, auc
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    roc_auc
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,p)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    Precision = (tp/(tp+fp))
    Recall = (tp/(tp+fn))
    print("tn: ",tn,"fp: ",fp,"fn:",fn,"tp: ",tp)
    print("Precision: ",Precision, "Recall :", Recall, "F1: ", ((2*Precision*Recall)/(Precision+Recall)))
    return a,b

def Tree(X_train, y_train, X_test, y_test):    
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    p = tree.predict(X_test)
    a = tree.score(X_train, y_train)
    b = tree.score(X_test, y_test)
    print("Default Decision Tree: training accuracy: ",a," testing accuracy: ",b)


    #roc curve
    y_prob = tree.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
    y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
    tree.score(X_test, y_pred)
    from sklearn.metrics import roc_curve, auc
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    roc_auc
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    from sklearn.metrics import confusion_matrix
    #confusion matrix
    cm = confusion_matrix(y_test,p)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    Precision = (tp/(tp+fp))
    Recall = (tp/(tp+fn))
    print("tn: ",tn,"fp: ",fp,"fn:",fn,"tp: ",tp)
    print("Precision: ",Precision, "Recall :", Recall, "F1: ", ((2*Precision*Recall)/(Precision+Recall)))
    return a,b

    
def NB(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()

    nb.fit(X_train, y_train)

    p = nb.predict(X_test)
    a = nb.score(X_train, y_train)
    b = nb.score(X_test, y_test)
    print("Deafult Naive Bayes: training accuracy: ",a," testing accuracy: ",b)



    #roc curve
    y_prob = nb.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  
    y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
    nb.score(X_test, y_pred)
    from sklearn.metrics import roc_curve, auc
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    roc_auc
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


    #confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,p)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    Precision = (tp/(tp+fp))
    Recall = (tp/(tp+fn))
    print("tn: ",tn,"fp: ",fp,"fn:",fn,"tp: ",tp)
    print("Precision: ",Precision, "Recall :", Recall, "F1: ", ((2*Precision*Recall)/(Precision+Recall)))
    return a,b
    
    
N = 1
Accu_Train_Reg = np.zeros(10)
Accu_Test_Reg = np.zeros(10)

Accu_Train_KNN = np.zeros(10)
Accu_Test_KNN = np.zeros(10)

Accu_Train_NN = np.zeros(10)
Accu_Test_NN = np.zeros(10)

Accu_Train_Tree = np.zeros(10)
Accu_Test_Tree = np.zeros(10)

Accu_Train_NB = np.zeros(10)
Accu_Test_NB = np.zeros(10)


while (N <= 10):
    X_train, X_test, y_train, y_test = readDATA(N) #given the number of feature will be used in training

    #logistic regression
    a,b = LogReg(X_train, y_train, X_test, y_test)
    Accu_Train_Reg[N-1] = a
    Accu_Test_Reg[N-1] = b
    #KNN
    c,d = KNN(X_train, y_train, X_test, y_test)
    Accu_Train_KNN[N-1] = c
    Accu_Test_KNN[N-1] = d
    #Neural Network
    e,f = NN(X_train, y_train, X_test, y_test)
    Accu_Train_NN[N-1] = e
    Accu_Test_NN[N-1] = f
    #Decision Tree
    g,h = Tree(X_train, y_train, X_test, y_test)
    Accu_Train_Tree[N-1] = g
    Accu_Test_Tree[N-1] = h
    #Naive Bayes
    i,l = NB(X_train, y_train, X_test, y_test)
    Accu_Train_NB[N-1] = i
    Accu_Test_NB[N-1] = l
    N += 1
    
print (Accu_Train_Reg)
print (Accu_Test_Reg)

print (Accu_Train_KNN)
print (Accu_Test_KNN)

print (Accu_Train_NN)
print (Accu_Test_NN)

print (Accu_Train_Tree)
print (Accu_Test_Tree)

print (Accu_Train_NB)
print (Accu_Test_NB)

