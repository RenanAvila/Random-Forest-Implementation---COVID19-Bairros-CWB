import pandas as pd
import numpy as np
import matplotlib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import ensemble
import matplotlib.pyplot as plt
import seaborn as sns

#Lendo nossa base de dados
data = pd.read_csv("bairros_dataset.csv")
data.head()

#definindo meu X e y
X = data[['Media Salarial','Nivel de Saneamento','Individuos P. Saude']]
y = data['Risco Propagação']

#Determinando partes para training e parte para teste
Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3)

#Aplicando Random Forest Classification
arvores = ensemble.RandomForestClassifier(n_estimators=21,max_depth=40,min_samples_leaf=3,random_state=10)
arvores.fit(Xtr, ytr)
y_prev= arvores.predict(Xval)

#Precisão da classificação
print("Precisão:",metrics.accuracy_score(yval,y_prev))

#Testando previsão
Xtest = [[760,0.2,50]]
print(arvores.predict(Xtest))
Xtest = [[3400,0.1,75]]
print(arvores.predict(Xtest))
Xtest = [[1200,0.09,90]]
print(arvores.predict(Xtest))
Xtest = [[2300,0.4,60]]
print(arvores.predict(Xtest))

#Grau de Importância dos recursos do bairros_dataset.csv para a classificação
features = ['Media Salarial','Nivel de Saneamento','Individuos P. Saude']
feature_imp = pd.Series(arvores.feature_importances_,index=features).sort_values(ascending=False)
# Criando barras
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()



