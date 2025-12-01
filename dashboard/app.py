
import streamlit as st
import pandas as pd
import numpy as np
from Model_Functions import Filtered_Data, Discretize, Probability_Matrix, Probability_Nsteps, Random_Walk, Plot_Random_Walk, Montecarlo_Simulation
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Markov Chains Engagement Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Markov Chains Engagement Dashboard")
st.markdown("""
This dashboard allows you to analyze and predict engagement levels using Markov chains. 
You can filter the data, visualize the Markov chain, and run Monte Carlo simulations.
""")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("dashboard/Final_Data.csv")
    data['Date_timestamp'] = pd.to_datetime(data['Date_timestamp'])
    return data

data = load_data()

# Sidebar for filters
st.sidebar.header("Filters")

# Allow user to select the number of filters
num_filters = st.sidebar.number_input("Select number of filters", min_value=0, max_value=5, value=1)

# Create a dictionary to store the filters
filters = {}
filter_cols = []

for i in range(num_filters):
    st.sidebar.markdown(f"**Filter {i+1}**")
    filter_column = st.sidebar.selectbox(f"Select column to filter by", data.columns, key=f"filter_col_{i}")
    
    if filter_column:
        unique_values = data[filter_column].unique()
        selected_value = st.sidebar.selectbox(f"Select value for {filter_column}", unique_values, key=f"filter_val_{i}")
        filters[filter_column] = selected_value
        filter_cols.append(filter_column)

# Apply filters
if filters:
    filtered_data = Filtered_Data(data, filter_cols, filters)
else:
    filtered_data = data.copy()

# Main content
st.header("Filtered Data")
st.dataframe(filtered_data)

# Engagement Over Time (Continuous)
st.header("Engagement Over Time (Continuous)")
if not filtered_data.empty:
    st.line_chart(filtered_data['Engagement'])
else:
    st.write("No data to display.")

# Markov Chain Analysis
st.header("Markov Chain Analysis")

# Discretize data
q1 = np.quantile(data['Engagement'], q=.25)
q2 = np.quantile(data['Engagement'], q=.50)
q3 = np.quantile(data['Engagement'], q=.75)
q4 = np.quantile(data['Engagement'], q=1)
quantiles = np.array([0, q1, q2, q3, q4])

discretized_data, labels = Discretize(filtered_data, quantiles)
st.write("Discretized Data (States):")
st.dataframe(discretized_data)

# Engagement State Over Time
st.header("Engagement State Over Time")
if not discretized_data.empty:
    # Map state labels to integers for plotting
    state_to_int = {label: i for i, label in enumerate(labels)}
    
    # Create a new column with integer states
    plot_data = discretized_data.copy()
    plot_data['State_Int'] = plot_data['State'].map(state_to_int)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=plot_data, x=plot_data.index, y='State_Int', ax=ax)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title('Engagement State Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Engagement State')
    st.pyplot(fig)
else:
    st.write("No data to display.")

# Probability Matrix
sparse_matrix, prob_matrix = Probability_Matrix(discretized_data, labels)

st.write("Transition Matrix:")
st.dataframe(pd.DataFrame(prob_matrix, index=labels, columns=labels))

# N-step transition
st.header("N-Step Transition Probability")
n_steps = st.slider("Select number of steps (n)", 1, 20, 1)
prob_n_steps = Probability_Nsteps(prob_matrix, n_steps)
st.write(f"{n_steps}-Step Transition Matrix:")
st.dataframe(pd.DataFrame(prob_n_steps, index=labels, columns=labels))

# Monte Carlo Simulation
st.header("Monte Carlo Simulation")
iterations = st.slider("Select number of iterations", 100, 10000, 500)

# Split data for simulation
train_size = int(len(discretized_data) * 0.8)
train_data = discretized_data.iloc[:train_size]
test_data = discretized_data.iloc[train_size:]

# Run simulation
sparse_matrix, prob_matrix = Probability_Matrix(train_data,labels)
acc_total, mean_acc = Montecarlo_Simulation(train_data, test_data, prob_matrix, labels, iterations)

st.write(f"Mean Accuracy of Monte Carlo Simulation: {mean_acc:.2f}")

# Plot accuracy distribution
st.write("Accuracy Distribution of Monte Carlo Simulation:")
fig, ax = plt.subplots()
sns.histplot(acc_total, kde=True, ax=ax)
st.pyplot(fig)

# Random Walk
st.header("Random Walk Simulation")
initial_state_options = {label: i for i, label in enumerate(labels)}
initial_state_label = st.selectbox("Select initial state for random walk", options=labels)
initial_state = initial_state_options[initial_state_label]
walk_steps = st.slider("Select number of steps for random walk", 10, 100, 50)

# Generate and plot random walk
random_walk_states = Random_Walk(prob_matrix, initial_state, walk_steps)
st.write("Simulated Random Walk:")
fig_walk, ax_walk = plt.subplots()
sns.lineplot(x=range(walk_steps + 1), y=random_walk_states, color="dodgerblue", ax=ax_walk)
ax_walk.set_yticks(range(len(labels)))
ax_walk.set_yticklabels(labels)
plt.xlabel("Day")
plt.ylabel("Engagement State")
st.pyplot(fig_walk)
