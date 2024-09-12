import pandas as pd
import numpy as np
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.svm import SVC
from sklearn.model_selection import train_test_split#,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings('always')
from warnings import simplefilter
from sklearn.exceptions import DataConversionWarning
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import warnings
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", UserWarning) 

df=pd.read_csv("phishing.csv")
print(df.head())
print(df.info())
df.isnull().sum()
X= df.drop(columns='class')
Y=df['class']
Y=pd.DataFrame(Y)
X.describe()

pd.value_counts(Y['class']).plot.bar()
plt.show()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.3,random_state=2)


X_norm = MinMaxScaler().fit_transform(X)

rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=10, step=10, verbose=5)
rfe_selector.fit(X_norm, Y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')
print(rfe_feature)

chi_selector = SelectKBest(chi2, k=10)
chi_selector.fit(X_norm, Y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')
print(chi_feature)

# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
# embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), '1.25*median')
# embeded_lr_selector.fit(X_norm, Y)
# embeded_lr_support = embeded_lr_selector.get_support()
# embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
# print(str(len(embeded_lr_feature)), 'selected features')
# print(embeded_lr_feature)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=25)
Y_sklearn = pca.fit_transform(X_norm)
cum_sum = pca.explained_variance_ratio_.cumsum()

pca.explained_variance_ratio_[:10].sum()

cum_sum = cum_sum*100

fig, ax = plt.subplots(figsize=(8,8))
plt.yticks(np.arange(0,110,10))
plt.bar(range(25), cum_sum, label='Cumulative _Sum_of_Explained _Varaince', color = 'b',alpha=0.5)
plt.title("Around 95% of variance is explained by the First 25 colmns ");
plt.show();


explained_variance=pca.explained_variance_ratio_
print(explained_variance.shape)
print(explained_variance.sum())
with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(25), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
plt.show()
    
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=20, n_iter=50, random_state=42)
svd.fit(X_norm)
explained_variance=svd.explained_variance_ratio_
print(explained_variance.shape)
print(svd.explained_variance_ratio_.sum())
with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(20), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
plt.show()    
    
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=-1,train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
              label="Cross-validation score")

    plt.legend(loc="best")
    return plt





''''RANDOM FOREST'''

print('RANDOM FOREST')
print('\n')
forest = RandomForestClassifier()
model6 = forest.fit(train_X,train_Y)
forest_predict = forest.predict(test_X)
acc_forest = accuracy_score(test_Y, forest_predict)
print('Random Forest Classifier Accuracy:',round(acc_forest*100,2))
print(classification_report(forest_predict,test_Y))
con  = confusion_matrix(forest_predict,test_Y)
plt.figure()
sns.heatmap(con,annot=True,fmt='.2f')
plt.show()
g = plot_learning_curve(model6," Random Forest Classifier learning curves",train_X,train_Y)
plt.show()


'''K-NEAREST NEIGHBORS'''

print('K-NEAREST NEIGHBORS')
print('\n')
knn=KNeighborsClassifier()
model7=knn.fit(train_X,train_Y)
knn_predict = knn.predict(test_X)
acc_knn = accuracy_score(test_Y, knn_predict)
print('K-Nearest Neighbour Accuracy:',round(acc_knn*100,2))
print(classification_report(knn_predict,test_Y))
con  = confusion_matrix(knn_predict,test_Y)
plt.figure()
sns.heatmap(con,annot=True,fmt='.2f')
plt.show()
g = plot_learning_curve(model7," KNN learning curves",train_X,train_Y)
plt.show()


