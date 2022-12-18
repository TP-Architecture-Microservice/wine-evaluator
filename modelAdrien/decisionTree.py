import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier




wine_data = pd.read_csv('Wines.csv')
wine_data = wine_data.drop('Unnamed: 13', axis=1)
wine_data = wine_data.drop('Id', axis=1)

fig = px.histogram(wine_data, x="quality", color="quality", marginal="box", hover_data=wine_data.columns)
# fig.show()

corr = wine_data.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

plt.show()

Y = wine_data['quality']
X = wine_data.drop('quality', axis=1)

X_features = X
X = StandardScaler().fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.15, random_state=None)

print('DecisionTreeClassifier')

model1 = DecisionTreeClassifier(random_state=1)
model1.fit(X_train, Y_train)
Y_pred1 = model1.predict(X_test)
print(classification_report(Y_test, Y_pred1))

print('RandomForestClassifier')

model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, Y_train)
Y_pred2 = model2.predict(X_test)
print(classification_report(Y_test, Y_pred2))
