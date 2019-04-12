# Import the necessary libraries first
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd
from pandas import read_excel
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
n = [1500,5000,7000,10000,20000]
my_sheet_name = 'transcripts-with-clinical-trans' 
xl_file = read_excel('prad_tcga_genes.xlsx',sheet_name = my_sheet_name)
xl2_file = read_excel('prad_tcga_clinical_data.xlsx')
y_data = xl_file.iloc[60486,1:]
x_data = xl_file.iloc[:60483,1:]
#y_data_target = y_data.iloc[3,1:].values

y_data[y_data == 6] = 0
y_data[y_data == 7] = 1
y_data[y_data == 8] = 1
y_data[y_data == 9] = 2
y_data[y_data == 10] = 2
fs = []
acc = []
df = pd.DataFrame(y_data)
df=df.astype('int')
genes_transpose = np.transpose(x_data)
for i in range(0,5):
    X_new = SelectKBest(chi2, k=n[i]).fit_transform(genes_transpose, df)
    classifier = SVC(kernel='poly')  
    X_train, X_test,y_train,y_test = train_test_split(X_new,df,test_size=0.3)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)      
    cnf_matrix.astype(float)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)   
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)        
    FP = FP.astype('float')                                      
    FN = FN.astype('float')          
    TP = TP.astype('float')            
    TN = TN.astype('float')          
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)    
    # Fall out or false positive rate
    FPR = FP/(FP+TN)       
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    print("For number of features = " + str(n[i]))
    print("Accuracy = ")
    print((ACC[0] + ACC[1] + ACC[2])/3)
    print("PPV = ")
    print((PPV[0] + PPV[1]+PPV[2])/3)
    print("NPV = ")
    print((NPV[0]+NPV[1]+NPV[2])/3)
    print("SENSITIVITY = ")
    print((sensitivity[0]+sensitivity[1]+sensitivity[2])/3)
    print("SPECIFICITY = ")
    print((specificity[0]+specificity[1]+specificity[2])/3)
    print("F-Score")
    print(f1_score(y_test, y_pred, average='micro'))
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    acc.append((ACC[0] + ACC[1] + ACC[2])/3)
    fs.append(f1_score(y_test, y_pred, average='micro'))
plt.subplots()
plt.bar(n,acc,width = 50)                       
plt.xlabel('Number of Features')           
plt.ylabel('Accuracy')  

plt.title('Accuracy vs Number of Features for svm')      
plt.show()
plt.subplots()
plt.bar(n,fs,width = 50)                       
plt.xlabel('F Score')           
plt.ylabel('Number of Features')  
plt.title('F score vs Number of Features for svm')      
plt.show()