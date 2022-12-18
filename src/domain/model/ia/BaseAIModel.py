#pour éviter les warnings intempestifs
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from abc import abstractmethod

from src.domain.model.ia.ModelDisplay import IAModelDescription, SerializedAIModel
from src.domain.prediction.Wine import WineQuality, Wine

#libs pour le model
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


class BaseAIModel:
    @abstractmethod
    #Y_test et Y_pred sont des listes de wine_quality. 
    # créé au moment de l'entrainement du modèle
    def describe(self, Y_test, Y_pred) -> IAModelDescription: 
        print(classification_report(Y_test, Y_pred, zero_division=1))


    @abstractmethod
    def serialize(self, model: RandomForestClassifier) -> SerializedAIModel: 
        joblib.dump(model, 'model.pkl')
        return SerializedAIModel("model.pkl")

    @abstractmethod
    def retrain(self) -> "BaseAIModel": 
        #load model
        loaded_model = joblib.load('model.pkl')
        wine_data = pd.read_csv('Wines.csv')
        wine_data = wine_data.drop('Unnamed: 13', axis=1)
        wine_data = wine_data.drop('Id', axis=1)
        
        Y = wine_data['quality']
        X = wine_data.drop('quality', axis=1)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.15, random_state=None)

        loaded_model = RandomForestClassifier(random_state=1)
        loaded_model.fit(X_train, Y_train)
        Y_pred = loaded_model.predict(X_test)
        #save model
        joblib.dump(loaded_model, 'model.pkl')
        
        

        


class QualityPredictorAIModel(BaseAIModel):
    @abstractmethod
    def predict(self, wine_caracteristic_values) -> WineQuality: 
        Wine_features=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
        test_valeurs=pd.DataFrame(columns=Wine_features)
        test_valeurs.loc[1]=wine_caracteristic_values
        #test_valeurs.loc[1]=[7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4]
        
        #return prediction with loaded model
        loaded_model = joblib.load('model.pkl')
        result = loaded_model.predict(test_valeurs)
        print(result)


class BestWinePredictorAIModel(BaseAIModel):
    @abstractmethod
    def predict(self) -> Wine: 
        print("Pour une note de 8, les features sont les suivantes")
        print("Le fixed acidity oscille entre 5.0 et 12.6")
        print("Le volatile acidity oscille entre 0.25 et 0.69" )
        print("Le citric acid oscille entre 0.05  et  0.72")
        print("Le residual sugar oscille entre 1.4  et  6.4")
        print("Le chlorides oscille entre 0.045  et  0.086")
        print("Le free sulfur dioxide oscille entre 3.0  et  42.0")
        print("Le total sulfur dioxide oscille entre 12.0  et  88.0")
        print("Le density oscille entre 0.9917  et  0.9988")
        print("Le pH oscille entre 2.88  et  3.72")
        print("Le sulphates oscille entre 0.63  et  1.1")
        print("Le alcohol oscille entre 9.8 et 14.0")
        print("Selon la HeatMap, les features qui ont le plus d'importance sont : fixed acidity, citric acidity, sulphates and alcohol")
        print("Ils faut maximiser la fixed acidity, les sulphates et l'alcool et minimiser la citric acidity")
        print("Selon nos résultats, le meilleur vin est le suivant :")
        print("fixed acidity : 12.6")
        print("sulphates : 1.1")    
        print("alcohol : 14.0")
        print("citric acidity : 0.05")
