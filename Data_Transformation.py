from datetime import datetime
from os import listdir
import pandas as pd
import logging as lg
import numpy as np


class dataTransform:

    def __init__(self):
        self.logger = lg

    def read_dataset(self):
        """Here we are reading the dataset"""
        try:
            lg.info("we are inside dataframe")
            df = pd.read_csv(r"L:\ML & AL\Projects\Mushroom Classification\Dataset\mushrooms.csv")
            print(df)
        except Exception as e:
            print("Check log for more info if your code fail")
            lg.error("error has occured")
            lg.exception(str(e))


#Calling the class
#obj=dataTransform()
#obj.read_dataset()

    def max_column(self):
        """With the help of class we can display max no. of columns"""
        try:
            lg.info("Expanding display to see more columns")
            df=pd.set_option('display.max_columns', None)
            print(df)
        except Exception as e:
            print("Check log for more info if your code fail")
            lg.error("error has occured")
            lg.exception(str(e))

#Calling the class
#obj=dataTransform()
#obj.max_column()

    def shape_dataset(self):
        """to know the shape of dataset"""
        try:
            lg.info("we are inside dataset")
            df=pd.read_csv(r"L:\ML & AL\Projects\Mushroom Classification\Dataset\mushrooms.csv")
            print("Number of rows", df.shape[0])
            print("Number of columns", df.shape[1])
        except Exception as e:
            print("Check log for info if your code fail")
            lg.error("error has occured")
            lg.exception(str(e))

#Calling the class
#obj=dataTransform()
#obj.shape_dataset()


    def null_values_check(self):
        """Check weather it contains null values or not"""
        df=pd.read_csv(r"L:\ML & AL\Projects\Mushroom Classification\Dataset\mushrooms.csv")
        try:
            lg.info("checking info of complete dataframe")
            for i in df:
                if i=='?':
                    print("it contains null values", i)
                else:
                    print("No null values in dataset", i)
        except Exception as e:
            print("Check log for info if your code fails")
            lg.error("error has occured")
            lg.exception(str(e))

#Calling the class
#obj=dataTransform()
#obj.null_values_check()

