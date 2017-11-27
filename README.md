# Final-Project
Before training, since the problem is a binary classification, I calculated the data distribution and found the class of instances were almost half-half, which is good.
Then features were ranked according to the evaluation of chi square value for each feature, chi square describes how important the feature is.
                    features  chi2 
0                 gill-color   5957.764469  
1                  ring-type   1950.610146  
2                  gill-size   1636.606833  
3                    bruises   1194.277352
4               gill-spacing    826.795274 
5                    habitat    751.309489 
6          spore-print-color    379.132729  
7                 population    311.766736
8   stalk-surface-above-ring    222.982400  
9                cap-surface    214.068544   
10  stalk-surface-below-ring    206.648180   


Firstly I tried training the data by using 5 algorithms (Logistic Regression, KNN, Neural Network, Decision Tree and Naive Bayes) with all 21 features (except for one feature with more than 1000 missing values, which is dropped). All 5 algorithms used were of their default setting, and gave very good accuracy of prediction (almost 100%). Results as following:

Default Logistic Regression: training accuracy:  1.0  testing accuracy:  1.0
[[2297    0]
 [   0  141]]
tn:  2297 fp:  0 fn: 0 tp:  141
Precision:  1.0 Recall:  1.0 F1:  1.0

KNN: training accuracy:  0.999824129441  testing accuracy:  0.998769483183
[[2297    0]
 [   3  138]]
tn:  2297 fp:  0 fn: 3 tp:  138
Precision:  1.0 Recall:  0.978723404255 F1:  0.989247311828

Default NN: training accuracy:  1.0  testing accuracy:  1.0
[[2297    0]
 [   0  141]]
tn:  2297 fp:  0 fn: 0 tp:  141
Precision:  1.0 Recall : 1.0 F1:  1.0

Default Decision Tree: training accuracy:  1.0  testing accuracy:  1.0
[[2297    0]
 [   0  141]]
tn:  2297 fp:  0 fn: 0 tp:  141
Precision:  1.0 Recall : 1.0 F1:  1.0

Deafult Naive Bayes: training accuracy:  1.0  testing accuracy:  1.0
[[2297    0]
 [   0  141]]
tn:  2297 fp:  0 fn: 0 tp:  141
Precision:  1.0 Recall : 1.0 F1:  1.0


As there is almost no space to improve the performance of models, the next step I did was trying to reduce the dimensions of data. Data was trained for 10 times by using 1 to 10 most important features according to the ranking obtained above. Plot of accuracy for each algorithms against number of important features was made, then I decide to use 6 most important features, because, 6 most important feature give fairly high accuracy and beyond which there is no significant increase of accuracy. 
With lower dimension of data, it is not possible to achieve 100% accuracy anymore. Therefore, considering the practical significance of the classfication, we want to choose a model that gives high predicion accuracy and low amount of false positive. Because false negative does not really hurt people but false positive might kill people. 
The final train I did was to train the data by 5 algorithms with 6 most important features for 100 times and I calculated the average of Accuracy and average of false positive. I found that, among 5 algorithms, the Decision Tree gives the lowest false positive, and interestingly, highest accuracy at the same time (~98.4%). 

Avg Accuracy Train LogReg:  0.935082320357
Avg Accuracy Train KNN:  0.980847822742
Avg Accuracy Train NN:  0.983682104939
Avg Accuracy Train Tree:  0.983786736421
Avg Accuracy Train NB:  0.92216802585
Avg Accuracy Test LogReg:  0.935082320357
Avg Accuracy Test KNN:  0.980847822742
Avg Accuracy Test NN:  0.983682104939
Avg Accuracy Test Tree:  0.983786736421
Avg Accuracy Test NB:  0.92216802585
Avg False Positive LogReg:  31.37
Avg False Positive KNN:  11.4
Avg False Positive NN:  6.66
Avg False Positive Tree:  6.45
Avg False Positive NB:  40.0










