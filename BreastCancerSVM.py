import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import pandas as pd

data = load_breast_cancer()
X = data.data  
y = data.target
feature_names = data.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#testing the best C and kernel
params = {
    'C': [0.1, 1, 10, 100] , 
    'kernel':['linear' , 'poly' , 'rbf' , 'sigmoid']
    }
grid = GridSearchCV(SVC(), params, cv=7,scoring='f1')
grid.fit(X_train, y_train)
results = pd.DataFrame(grid.cv_results_)
# print(results[['param_kernel', 'param_C', 'mean_test_score', 'std_test_score', 'rank_test_score']])
# print("Best C:", grid.best_params_['C'])
# print("Best kernel:", grid.best_params_['kernel'])


#getiing the best no. of featuers
svc = SVC(kernel='linear' ,  C =  grid.best_params_['C'])
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
rfecv.fit(X_train, y_train)
# print(f"عدد الميزات المثلى: {rfecv.n_features_}")
# print(f"الميزات المختارة (True = تم اختيارها):\n{rfecv.support_}")
# print(f"ترتيب الميزات:\n{rfecv.ranking_}")


# SVC model
model = SVC(kernel= #grid.best_params_['kernel']
                    'linear',
                    C = grid.best_params_['C']) 

# masking only the features that we get from the RFECV
X_train = X_train[:, rfecv.support_]
X_test= X_test[:, rfecv.support_]
model.fit(X_train, y_train)

# pred :
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ploting decision boundary
    
# PCA فقط للرسم (لا تستخدمها في تدريب النموذج)
pca = PCA(n_components=3)
pca.fit(X_train)  # تتعلم PCA من بيانات التدريب
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# نأخذ متجه الأوزان الأصلي من النموذج (في الفضاء بعد اختيار الميزات)
w_orig = model.coef_[0]  # في بعد الميزات المختارة
b = model.intercept_[0]

# نحصل على مكونات PCA (المصفوفة P)
P = pca.components_  # شكلها (3, n_features_after_RFECV)

# نحسب الإسقاط إلى 3D:
w_3D = P @ w_orig  # (3,) الناتج

# شبكة من النقاط للرسم
xx, yy = np.meshgrid(
    np.linspace(X_test_pca[:, 0].min(), X_test_pca[:, 0].max(), 50),
    np.linspace(X_test_pca[:, 1].min(), X_test_pca[:, 1].max(), 50)
)

zz = (-w_3D[0]*xx - w_3D[1]*yy - b) / w_3D[2]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2], c=y_test, cmap='bwr', alpha=0.6)

ax.plot_surface(xx, yy, zz, alpha=0.3, color='green')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('Original Model Decision Surface in PCA 3D Space')
plt.show()