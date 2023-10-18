import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.decomposition import PCA
import logging as lg


class GradientBoosting:

    def __init__(self):
        self.logger = lg

    def gbc(self):
        """To create Gradient Boosting classifier"""
        try:
            lg.info("inside gradient boosting classifier")
            df = pd.read_csv(r"L:\ML & AL\Projects\Mushroom Classification\Dataset\encoder_mushroom_file.csv")
            X = df.drop('class', axis=1)
            Y = df['class']

            pca1 = PCA(n_components=7)
            pca_fit = pca1.fit_transform(X)

            X_train, X_test, Y_train, Y_test = train_test_split(pca_fit, Y, test_size=0.20, random_state=42)

            gb = GradientBoostingClassifier()
            gb.fit(X_train,Y_train)
            Y_predict6 = gb.predict(X_test)

            print("Accuracy of RandomForest", accuracy_score(Y_test, Y_predict6))

            gbc_model = GradientBoostingClassifier()
            gbc_model.fit(pca_fit, Y)
            filename = r'Model/gbcPickle.pkl'
            pickle.dump(gbc_model, open(filename, 'wb'))

        except Exception as e:
            print("check log for info if your code fails")
            lg.error("error has occured")
            lg.exception(str(e))




