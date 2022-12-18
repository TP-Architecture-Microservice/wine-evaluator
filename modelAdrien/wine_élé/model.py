import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import csv
from joblib import dump, load

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
def save_model(Wine_ridge:sklearn.linear_model._ridge.Ridge):
    """Sauvegarde le modèle ridge

    Parameters
    ----------
    Wine_ridge : sklearn.linear_model._ridge.Ridge
        Le modèle ridge

    """
    dump(Wine_ridge, 'Wine_ridge.modele')

def load_model()->sklearn.linear_model._ridge.Ridge:
    """Récupère le modèle déja entrainé

    Returns
    -------
    sklearn.linear_model._ridge.Ridge
        retourne le modèle déja entrainé
    """
    return load('Wine_ridge.modele')

def ridge_model():
    """Fait ridge

    Returns
    -------
    sklearn.linear_model._ridge.RidgeCV
        le modele ridge
    DataFrame
        x_test
    DataFrame 
        y_test
    """
    X,y=get_wine_features_and_wine_quality_separately()
    #Separer en train et test avec une taille de test de 0.3
    x_train,x_test,y_train,y_test = train_test_split(X,y,random_state = 2, test_size = 0.3)
    #alpha=0.3 est le meilleur alpha
    Wine_ridge = RidgeCV(alphas = 0.3)
    Wine_ridge.fit(x_train,y_train)

    save_model(Wine_ridge)
    return (Wine_ridge,x_test,y_test)

def ridge_model_predictions()-> np.ndarray :
    """Récupère la prédiction ridge

    Returns
    -------
    np.ndarray
        Les qualités des vins du x_test
    """
    Wine_ridge,x_test,y_test=ridge_model();    
    return  Wine_ridge.predict(x_test)



def mse_model(y_test : pd.core.series.Series,ridge_predict : np.ndarray )->float:
    """Récupère le modèle mse

    Parameters
    ----------
    y_test : pd.core.series.Series
        les qualités des vins du test
    ridge_predict : np.ndarray
        Les qualités prédites par ridge des vins du test

    Returns
    -------
    float
        La valeur du mse,Plus le mse est petit plus le modèle est bon
    """
    #print(accuracy_score(y_test, ridge_predict, normalize=False))
    return mean_squared_error(y_test,ridge_predict)


def get_the_params():
    """Récupère les paramètres de ridge

    """
    print("Les paramètres sont :  alpha :",ridge_model()[0].alpha_)
    print("La valeur mse est ",mse_model(ridge_model()[2],ridge_model_predictions()))


def ridge_model_predict_one_value(wine_caracteristic_values : list) -> float:
    """Récupère la prédiction de la valeur mise en entrée

    Parameters
    ----------
    wine_caracteristic_values : list
        Un vin avec toutes ses caractèristiques sauf l'id

    Returns
    -------
    float
        La qualité du vin en entrée
    """
    X,y=get_wine_features_and_wine_quality_separately()
    Wine_features=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    test_valeurs=pd.DataFrame(columns=Wine_features)
    test_valeurs.loc[1]=wine_caracteristic_values
    #test_valeurs.loc[1]=[7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4]
    x_train,x_test,y_train,y_test = train_test_split(X,y,random_state = 2, test_size = 0.3)
    Wine_ridge = Ridge(alpha = 0.3)
    Wine_ridge.fit(x_train,y_train)
    return (Wine_ridge.predict(test_valeurs))



def get_all_values(j:int)->np.ndarray:
    """Récupère toutes les valeurs qui ont la même qualité de vin

    Parameters
    ----------
    j : int
        La qualité 

    Returns
    -------
    np.ndarray
        Tous les vins ayant comme qualité le chiffre passé en entrée
    """
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
    """Récupère la meilleur qualité du fichier Wines.csv

    Returns
    -------
    int
        la meilleure qualité
    """
    data=pd.read_csv('Wines.csv')
    y = data['quality']
    max_quality=0
    for i in range(len(y)):
        if(y[i]>max_quality):
            max_quality=y[i]
    return max_quality

def get_the_best_wine():
    """Affiche les valeurs qui font qu'un vin a une haute qualité
    """
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
   


def write_new_wine_in_csv(wine_caracteristic_values : list):
    """Rajoute un vin dans le csv

    Parameters
    ----------
    wine_caracteristic_values : list
        Liste des caractéristiques d'un vin 
    """
    with open('Wines.csv','a',newline='',encoding='utf-8') as toto:
        writer=csv.writer(toto)
        writer.writerow(wine_caracteristic_values)


def ridge_model_predict_one_value_with_load_model(wine_caracteristic_values : list) -> float:
    """Fait la prédiction ridge avec le modèle sauvegarder

    Parameters
    ----------
    wine_caracteristic_values : list
        Un vin avec toutes ses caractèristiques sauf l'id

    Returns
    -------
    float
        La qualité du vin
    """
    X,y=get_wine_features_and_wine_quality_separately()
    Wine_features=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    test_valeurs=pd.DataFrame(columns=Wine_features)
    test_valeurs.loc[1]=wine_caracteristic_values
    #test_valeurs.loc[1]=[7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4]
    
    return (load_model().predict(test_valeurs))

