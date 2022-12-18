def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import joblib  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



def reading_csv():
    wine_data = pd.read_csv('Wines.csv')
    wine_data = wine_data.drop('Unnamed: 13', axis=1)
    wine_data = wine_data.drop('Id', axis=1)
    return wine_data

def get_wine_features_and_wine_quality_separately():
    """Récupère les features des vins.
    Returns
    -------
    Les features d'un coté et la qualité de chaque vin dans deux variables séparés
    """

    #Recuperer les donnees
    data=pd.read_csv('Wines.csv')
    #Recuperer X
    Wine_features=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    X = data[Wine_features]
    #Recuperer Y
    y = data['quality']
    return (X,y)

def print_heatmap(wine_data):
    corr = wine_data.corr()
    plt.subplots(figsize=(15,10))
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
    plt.show()




def split_data_and_predict(wine_data):
    Y = wine_data['quality']
    X = wine_data.drop('quality', axis=1)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.15, random_state=None)

    print('RandomForestClassifier')

    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    save_model(model)
    #ça print toutes les métriques
    print(classification_report(Y_test, Y_pred, zero_division=1))


#tentative de load sur la save, mais quand je print les résulstats, j'ai 95% d'accuracy, ce qui est faux, mais je sais pas d'où ça vient
def split_data_from_load(wine_data):
    YY = wine_data['quality']
    XX = wine_data.drop('quality', axis=1)
    XX = StandardScaler().fit_transform(XX)
    X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size=.15)

    print('Loaded')

    model = load_model()
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, zero_division=1))



def save_model(model):
    """Sauvegarde le modèle ridge

    Parameters
    ----------
    Wine_ridge : sklearn.linear_model._ridge.Ridge
        Le modèle ridge

    """
    joblib.dump(model, 'save.model')
    
    
def load_model():
    """Récupère le modèle déja entrainé

    Returns
    -------
    sklearn.linear_model._ridge.Ridge
        retourne le modèle déja entrainé
    """
    return joblib.load('save.model')


#récupère toutes les valeurs
def get_all_values(j:int)->np.ndarray:

    Wine_features=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    Good_value=pd.DataFrame(columns=Wine_features)
    data=pd.read_csv('Wines.csv')
    iterations=0
    y = data['quality']
    for i in range(len(y)):
        if(y[i]==j):
            Good_value.loc[iterations]=data.loc[i]
            iterations+=1
    return(Good_value)


def get_the_best_quality()->int:

    data=pd.read_csv('Wines.csv')
    y = data['quality']
    max_quality=0
    for i in range(len(y)):
        if(y[i]>max_quality):
            max_quality=y[i]
    return max_quality


#je l'ai changé dans BaseModelAI, ça print les classes importantes, et donne la meilleure selon nous
def get_the_best_wine():

    all_values=get_all_values(get_the_best_quality())
    #qualité lie avec alcool sulphates,citric acid
    print("Pour une note de ",get_the_best_quality(),"les features sont les suivantes")
    print("Le fixed acidity oscille entre",all_values['fixed acidity'].min()," et ",all_values['fixed acidity'].max())
    print("Le volatile acidity oscille entre",all_values['volatile acidity'].min()," et ",all_values['volatile acidity'].max())
    print("Le citric acid oscille entre",all_values['citric acid'].min()," et ",all_values['citric acid'].max())
    print("Le residual sugar oscille entre",all_values['residual sugar'].min()," et ",all_values['residual sugar'].max())
    print("Le chlorides oscille entre",all_values['chlorides'].min()," et ",all_values['chlorides'].max())
    print("Le free sulfur dioxide oscille entre",all_values['free sulfur dioxide'].min()," et ",all_values['free sulfur dioxide'].max())
    print("Le total sulfur dioxide oscille entre",all_values['total sulfur dioxide'].min()," et ",all_values['total sulfur dioxide'].max())
    print("Le density oscille entre",all_values['density'].min()," et ",all_values['density'].max())
    print("Le pH oscille entre",all_values['pH'].min()," et ",all_values['pH'].max())
    print("Le sulphates oscille entre",all_values['sulphates'].min()," et ",all_values['sulphates'].max())
    print("Le alcohol oscille entre",all_values['alcohol'].min()," et ",all_values['alcohol'].max())
    print("Selon la HeatMap, les features qui ont le plus d'importance sont : fixed acidity, citric acidity, sulphates and alcohol")
    print("Ils faut maximiser la fixed acidity, les sulphates et l'alcool et minimiser la citric acidity")
   


def write_new_wine_in_csv(wine_caracteristic_values : list):

    with open('Wines.csv','a',newline='',encoding='utf-8') as toto:
        writer=csv.writer(toto)
        writer.writerow(wine_caracteristic_values)


#on file en entrée une liste wine_caracteristic_values, qui contient les valeurs d'un seul vin
def random_forest_model_predict_one_value_with_load_model():

    Wine_features=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    test_valeurs=pd.DataFrame(columns=Wine_features)
    #test_valeurs.loc[1]=wine_caracteristic_values
    test_valeurs.loc[1]=[7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4]
    
    return (load_model().predict(test_valeurs))




if __name__ == '__main__':
    wine_data = reading_csv()
    print_heatmap(wine_data)
    split_data_and_predict(wine_data)
    print('now load and run')
    split_data_from_load(wine_data)
    print(random_forest_model_predict_one_value_with_load_model())
    get_the_best_wine()
    