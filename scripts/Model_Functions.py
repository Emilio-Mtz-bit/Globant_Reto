import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

def Filtered_Data(data, group_cols, filters = None):
    """
    Groups by Date_timestamp and specified columns, calculates median Engagement.
    Allows filtering specific values *before* calculation.
    """
    df = data.copy()
    
    if filters:
        for col, value in filters.items():
            if isinstance(value, list):
                df = df[df[col].isin(value)]
            else:
                df = df[df[col] == value]
    
    date_col = ["Date_timestamp"]
    group_keys = date_col + group_cols
    result = (df.groupby(group_keys)["Engagement"]
                .median()
                .reset_index()
                .set_index('Date_timestamp')
                .sort_index())

    return result

    
    


def Discretize(data, quantiles):
    """
    This function discretize the engagment column putin into buckets depending on the range, so it can be used later in as states in a Markov Chain Model

    """
    
    df = data.copy()
    
    bins = quantiles
    labels = [f"{quantiles[0]}-{quantiles[1]}", f"{quantiles[1]}-{quantiles[2]}", f"{quantiles[2]}-{quantiles[3]}",f"{quantiles[3]}-{quantiles[4]}"]
    df['State'] = pd.cut(df['Engagement'], bins=bins, labels=labels, right=True, include_lowest=True)

    return df , np.array(labels)
    
    
def Probability_Matrix(df, labels):
    """
    This function count the number of times it passes from one state to anther and then uses maximun likelihood to estimate the probabilities
    """
    sparse_matrix = np.zeros((labels.shape[0], labels.shape[0]))
    labels = labels
    label_to_index = {label: i for i, label in enumerate(labels)}
    for i in range(df.shape[0]-1):
        sparse_matrix[label_to_index[df['State'].iloc[i]]][label_to_index[df['State'].iloc[i+1]]] += 1
    
    row_sums = sparse_matrix.sum(axis=1, keepdims=True)
    probability_matrix = sparse_matrix/row_sums
    
    probability_matrix = np.nan_to_num(probability_matrix, nan=0.0)
        
    return sparse_matrix, probability_matrix

def Probability_Nsteps(prob_matrix, n):
    """
    This function calculates the probability matrix in n-steps
    """
    
    P_n = np.linalg.matrix_power(prob_matrix, n)
    
    return P_n
        
         
    
def Random_Walk(prob_matrix, initial_state, n):
    """
    This function simulates the random walk of the markov chain
    """  
    x = np.array([initial_state])
    states = np.array([0,1,2,3])
    for i in range(n):
        x = np.append(x, np.random.choice(states, size=1, replace=False, p = prob_matrix[x[i],:]))
    
    return x

def Plot_Random_Walk(x, label):
    n = len(x) - 1
    plt.figure(figsize = (7, 5))
    g = sns.lineplot(x = range(n + 1), y = x, color = "dodgerblue")
    g.set_yticks([0, 1, 2, 3])
    g.set_yticklabels(label)
    plt.xlabel("Dia", fontsize = 14)
    plt.ylabel("Engagement", fontsize = 14)
    plt.show();
    
  
    
def Montecarlo_Simulation(data_train,data_test, prob_matrix,labels, iterations):
    
    n = len(data_test)
    
    label_to_index = {label: i for i, label in enumerate(labels)}
    initial_state = label_to_index[data_train['State'].iloc[-1]]
    
    acc_total = []
    for i in range(iterations):
        
        state_pred = Random_Walk(prob_matrix,initial_state,n)
        
            
        y_true = data_test['State'].map(label_to_index)
        y_pred = state_pred[1:]
        
        acc = accuracy_score(y_true, y_pred)
        acc_total.append(acc)
        
        
    return acc_total, np.mean(acc_total)
    
    
