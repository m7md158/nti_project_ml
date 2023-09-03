import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score,precision_score, f1_score
from sklearn.impute import SimpleImputer


df = pd.read_csv('heart_disease_uci.csv')



# fillna data with mean
# df['trestbps'].fillna(value= df['trestbps'].mean(), inplace=True)
# df['chol'].fillna(value= df['chol'].mean(), inplace=True)
# df['thalch'].fillna(value= df['thalch'].mean(), inplace=True)
# df['oldpeak'].fillna(value= df['oldpeak'].mean(), inplace=True)
# df['ca'].fillna(value= df['ca'].mean(), inplace=True)


x = df.iloc[:, 0:15].values
y = df.iloc[:, 15].values




# lable encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])
x[:, 7] = le.fit_transform(x[:, 7])
x[:, 10] = le.fit_transform(x[:, 10])


# one hot encoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transformer = ColumnTransformer(transformers=[('onehot', OneHotEncoder(), [3, 4, 8, 12, 14])], remainder='passthrough')
x = transformer.fit_transform(x)



# fill in the nan values with mean
from sklearn.impute import SimpleImputer
import numpy as np
simple = SimpleImputer(missing_values=np.nan ,strategy='mean')
x = simple.fit_transform(x)


# Data division
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= .3, random_state=0)




# Random forest
def Rondom_fores():
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(x_train,y_train)
    y_pred_rm = rf.predict(x_test)

    # Evaluation
    print('accuracy score is : ', accuracy_score(y_test,y_pred_rm))
    print('the precision is : ', precision_score(y_test,y_pred_rm, average='macro'))
    # print('the confuion_matrix is : ', confusion_matrix(y_test,y_pred_rm))
    print('recall is : ', recall_score(y_test,y_pred_rm,average='weighted'))
    print('the f1_score is : ', f1_score(y_test,y_pred_rm, average='micro'))




# KNN
def Knn():
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    # Evaluation
    print('accuracy score is : ', accuracy_score(y_test,y_pred))
    print('the precision is : ', precision_score(y_test,y_pred, average='macro'))
    # print('the confuion_matrix is : ', confusion_matrix(y_test,y_pred))
    print('recall is : ', recall_score(y_test,y_pred,average='weighted'))
    print('the f1_score is : ', f1_score(y_test,y_pred, average='micro'))





# DecisionTree
def Decisiontree():
    from sklearn.tree import DecisionTreeClassifier
    dc = DecisionTreeClassifier()
    dc.fit(x_train, y_train)
    y_pred_dc = dc.predict(x_test)

    # Evaluation
    print('accuracy score is : ', accuracy_score(y_test,y_pred_dc))
    print('the precision is : ', precision_score(y_test,y_pred_dc, average='macro'))
    # print('the confuion_matrix is : ', confusion_matrix(y_test,y_pred_dc))
    print('recall is : ', recall_score(y_test,y_pred_dc,average='weighted'))
    print('the f1_score is : ', f1_score(y_test,y_pred_dc, average='micro'))




# SVM _ linear
def SVM_linear():
    from sklearn.svm import SVC
    svm_linear = SVC(kernel='linear')
    svm_linear.fit(x_train, y_train)
    y_pred_svm_linear = svm_linear.predict(x_test)

    # Evaluation
    print('accuracy score is : ', accuracy_score(y_test,y_pred_svm_linear))
    print('the precision is : ', precision_score(y_test,y_pred_svm_linear, average='macro'))
    # print('the confuion_matrix is : ', confusion_matrix(y_test,y_pred_svm_linear))
    print('recall is : ', recall_score(y_test,y_pred_svm_linear,average='weighted'))
    print('the f1_score is : ', f1_score(y_test,y_pred_svm_linear, average='micro'))





# SVM_ rbf
def SVM_rbf(): 
    from sklearn.svm import SVC
    svm_linear = SVC(kernel='rbf')
    svm_linear.fit(x_train, y_train)
    y_pred_svm_rbf = svm_linear.predict(x_test)

    # Evaluation
    print('accuracy score is : ', accuracy_score(y_test,y_pred_svm_rbf))
    print('the precision is : ', precision_score(y_test,y_pred_svm_rbf, average='macro'))
    # print('the confuion_matrix is : ', confusion_matrix(y_test,y_pred_svm_rbf))
    print('recall is : ', recall_score(y_test,y_pred_svm_rbf,average='weighted'))
    print('the f1_score is : ', f1_score(y_test,y_pred_svm_rbf, average='micro'))






# Naive_bayes
def Naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    gc = GaussianNB()
    gc.fit(x_train, y_train)
    y_pred_gc = gc.predict(x_test)

    # Evaluation
    print('accuracy score is : ', accuracy_score(y_test,y_pred_gc))
    print('the precision is : ', precision_score(y_test,y_pred_gc, average='macro'))
    # print('the confuion_matrix is : ', confusion_matrix(y_test,y_pred_gc))
    print('recall is : ', recall_score(y_test,y_pred_gc,average='weighted'))
    print('the f1_score is : ', f1_score(y_test,y_pred_gc, average='micro'))







def Cross_valdation():
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    sv = SVC(kernel='linear')
    score = cross_val_score(estimator= sv, X= x, y=y, cv=10)
    print(score)
    print(score.mean())




while True:
        selection = input("\n \n '1' =>  Random Forest \n '2' =>  KNN \n '3' =>  Decision Tree \n '4' =>  SVM_Linear \n '5' =>  SVM_rbf \n '6' =>  Naive bays \n '8' =>  Cross_valdation\n " )
        
        if selection == "1":
             Rondom_fores()
        
        elif selection == '2':
             Knn()

        elif selection == '3':
             Decisiontree()

        elif selection == '4':
             SVM_linear()
        
        elif selection =='5':
             SVM_rbf()
    
        elif selection == '6':
             Naive_bayes()
        
        elif selection == '8':
             Cross_valdation()

        else:
             print('wrong please try again')
        
