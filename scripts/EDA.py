import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


## Lectura de datos
data = pd.read_csv("../data/data_globant.csv")
print(f"Columnas en el data set: {data.columns}")

## Informacion general de la data
data.drop(columns=['Email','Email Leader'], inplace=True)
data.info()
data_num = data.select_dtypes(include='number')
data_obj = data.select_dtypes(include='object')
data_obj


prueba = data.groupby(['Name'])['Team Name'].nunique()
prueba[prueba>1]