import numpy as np
import pandas as pd

def Filtered_Data(data, column):
    date = ["Date_timestamp"]
    column_groupby = date + column
    data_filter = data.groupby(column_groupby)["Engagement"].median().reset_index()
    data_filter = data_filter.set_index('Date_timestamp').sort_index()
    return data_filter
    
    
def Filtered_Data2(df, filtros_exactos):
    
    # Inicializar la máscara booleana como True (para empezar a combinar)
    mascara = pd.Series([True] * len(df), index=df.index)
    
    for columna, valor in filtros_exactos.items():
        # Aplicar la condición y combinarla con la máscara existente usando AND (&)
        mascara = mascara & (df[columna] == valor)
        
    df_filtrado = df[mascara]
    return df_filtrado



def Discretize(df):
    bins = [0,1,2,3,4,5]
    labels = ["0-1", "1-2", "2-3","3-4",'4-5']
    df['State'] = pd.cut(df['Engagement'], bins=bins, labels=labels, right=True)
    
    
def Probability_Matrix(df):
    sparse_matrix = np.zeros((5, 5))
    labels = ["0-1", "1-2", "2-3","3-4",'4-5']
    label_to_index = {label: i for i, label in enumerate(labels)}
    for i in range(df.shape[0]-1):
        sparse_matrix[label_to_index[df['State'].iloc[i]]][label_to_index[df['State'].iloc[i+1]]] += 1
    
    row_sums = sparse_matrix.sum(axis=1, keepdims=True)
    probability_matrix = sparse_matrix/row_sums
    
    probability_matrix = np.nan_to_num(probability_matrix, nan=0.0)
        
    return sparse_matrix, probability_matrix
        
         
    
    

