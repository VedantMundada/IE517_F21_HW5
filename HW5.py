import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
print( 'The scikit learn version is {}.'.format(sklearn.__version__))
path = "C:/Users/Vedant/Desktop/UIUC/SEM 1/Machine Learning/Week 5/hw5_treasury yield curve data.csv"
treasury = pd.read_csv(path)
print(treasury.keys())
del(treasury['Date'])
cols=['SVENF01', 'SVENF02', 'SVENF03', 'SVENF04', 'SVENF05', 'SVENF06',
       'SVENF07', 'SVENF08', 'SVENF09', 'SVENF10', 'SVENF11', 'SVENF12',
       'SVENF13', 'SVENF14', 'SVENF15', 'SVENF16', 'SVENF17', 'SVENF18',
       'SVENF19', 'SVENF20', 'SVENF21', 'SVENF22', 'SVENF23', 'SVENF24',
       'SVENF25', 'SVENF26', 'SVENF27', 'SVENF28', 'SVENF29', 'SVENF30',
       'Adj_Close']
# sns.pairplot(treasury[cols],height=2.5)
# plt.tight_layout()
# plt.show()

corr=treasury[cols].corr()
print(corr)
sns.set(font_scale=0.65)
sns.heatmap(corr)
plt.show()
print(("Number of Rows of Data = " + str(len(treasury)) + '\n'))
n_rows=len(treasury)
n_col=0
for column in treasury.values[0,:]:
    n_col=n_col+1
print("Number of columns of Data = " , n_col , '\n')
col_summ=treasury.describe()
print("The summary for each column is \n",col_summ)
sns.set(font_scale=1)
plt.plot(col_summ.values[1,:])
plt.xlabel("Attribute number")
plt.ylabel("Mean Values of the instances")
plt.title(("Means of attributes"))
plt.show()

sns.set(font_scale=0.9)
plt.plot(cols[:30],corr.values[30,0:30])
plt.xticks(rotation=90)
plt.xlabel("Attribute")
plt.ylabel("Correlation Coefficient with Adj_Close")
plt.title(("Corealtion with Adj_Close"))
plt.show()


#%%

sc=StandardScaler()
y=treasury['Adj_Close'].values
X=treasury.values[:,:31]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=42)

print("Training set X ", X_train)
print("Training set y ", y_train)

X_train_std=sc.fit_transform(X_train)
X_test_std=sc.transform(X_test)
print("Standardized training set ",X_train_std)
print("Standardized test set ",X_test_std)



#%%
lr=LinearRegression()
lr.fit(X_train_std,y_train)
y_pred=lr.predict(X_test_std)
lr_acc_test=lr.score(X_test_std,(y_test))
lr_acc_train=lr.score(X_train_std,(y_train))
print("The training accuracy for Linear Regression is ",lr_acc_train)
print("The test accuracy for Linear Regression is ",lr_acc_test)
print("The MSE is (Linear Rgression )", MSE(y_test,y_pred))


#%%
from sklearn.svm import SVR
svm = SVR(kernel='linear')
svm.fit(X_train_std,y_train)
y_pred_svm = svm.predict(X_test_std)
print("The training accuracy for SVM Regression(linear) is ",svm.score(X_train_std,(y_train)))
print("The test accuracy for SVM Regression(linear) is ",svm.score(X_test_std,y_test))
print("The MSE (linear SVM)is ", MSE(y_test,y_pred_svm))
svm = SVR(kernel='linear')

svm_rbf = SVR()
svm_rbf.fit(X_train_std,y_train)
y_pred_svm = svm_rbf.predict(X_test_std)
print("The training accuracy for SVM Regression(RBF) is ",svm_rbf.score(X_train_std,(y_train)))
print("The test accuracy for SVM Regression(RBF) is ",svm_rbf.score(X_test_std,y_test))
print("The MSE is ", MSE(y_test,y_pred_svm))
#%%
from sklearn.decomposition import PCA
pca=PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
exp_var_pca= pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.title("Cumulative and histogram plot for explained variance ratio")
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#%%
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
X_train_pca3 = pca.fit_transform(X_train_std)
X_test_pca3 = pca.transform(X_test_std)

print("The three principal components account for ",sum(pca.explained_variance_ratio_)*100,"% of the total data")
lr_pca=LinearRegression()
lr_pca.fit(X_train_pca3,y_train) 
y_pred_lr_pca=lr_pca.predict(X_test_pca3)

lr_acc_train_pca=lr_pca.score(X_train_pca3,y_train)
lr_acc_test_pca=lr_pca.score(X_test_pca3,y_test)
print("The training accuracy for Linear Regression is (post PCA data) ",lr_acc_train_pca)
print("The test accuracy for Linear Regression is (post PCA data) ",lr_acc_test_pca)
print("The MSE is ", MSE(y_test,y_pred_lr_pca))
    
svm_pca= SVR(kernel='linear')
svm_pca.fit(X_train_pca3,y_train)
y_pred_svm_pca = svm_pca.predict(X_test_pca3)
print("The training accuracy for SVM Regression (linear) is (post PCA data)",svm_pca.score(X_train_pca3,(y_train)))
print("The test accuracy for SVM Regression (linear) is (post PCA data)",svm_pca.score(X_test_pca3,y_test))
print("The MSE is ", MSE(y_test,y_pred_svm_pca))

#%%
svm_pca= SVR()
svm_pca.fit(X_train_pca3,y_train)
y_pred_svm_pca = svm_pca.predict(X_test_pca3)
print("The training accuracy for SVM Regression (RBF) is (post PCA data)",svm_pca.score(X_train_pca3,(y_train)))
print("The test accuracy for SVM Regression (RBF)  is (post PCA data)",svm_pca.score(X_test_pca3,y_test))
print("The MSE is ", MSE(y_test,y_pred_svm_pca))

#%%

print("My name is Vedant Mundada")
print("My NetID is: vkm3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")




