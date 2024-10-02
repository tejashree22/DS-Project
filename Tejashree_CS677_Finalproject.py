#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import calendar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[93]:


import datetime as dt

import plotly.io as pio
pio.templates


# In[94]:



import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from IPython.display import HTML


# In[95]:


file_url = 'https://docs.google.com/spreadsheets/d/1JCfb21vzJBgGPMVE-7YCaSIzLVPCi_tkiFT0jFcDzNE/export?format=csv'
data = pd.read_csv(file_url)


# In[96]:


data.head(7)


# In[97]:


def dataset_info(data):
    # Print the shape of the DataFrame (number of rows and columns)
    print(f"DataFrame Shape: {data.shape}\n")

    # Iterate over each column in the DataFrame
    print("Column Information:")
    for col in data.columns:
        # Print the name of the column
        print(f"Column: {col}")
        
        # Print the data type of the column
        print(f"  - Data Type: {data[col].dtype}")
        
        # Print the count of non-null (non-missing) values in the column
        print(f"  - Non-Null Count: {data[col].notnull().sum()}")
        
        # Print the number of unique values in the column
        print(f"  - Unique Values: {data[col].nunique()}")
        
        # Print the first few values of the column for a quick preview
        print("  - First Few Values:", data[col].head().values, "\n")
        
dataset_info(data)



# In[98]:


missing_values = data.isnull().sum()

print(missing_values)



# In[99]:



original_columns = data.columns.tolist()

# Now, create a mapping from old column names to new column names
column_mapping = {
    original_columns[0]: 'States',
    original_columns[1]: 'Date',
    original_columns[2]: 'Frequency',
    original_columns[3]: 'Estimated Unemployment Rate',
    original_columns[4]: 'Estimated Employed',
    original_columns[5]: 'Estimated Labour Participation Rate',
    original_columns[6]: 'Region',
    original_columns[7]: 'longitude',
    original_columns[8]: 'latitude'
}

# Rename the columns using the mapping
data = data.rename(columns=column_mapping)


# In[100]:


# For example, if your date format is 'DD-MM-YYYY', you can use '%d-%m-%Y'
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')


# In[101]:


import pandas as pd
import calendar

# Convert 'Frequency' and 'Region' to categorical data types
data['Frequency'] = data['Frequency'].astype('category')
data['Region'] = data['Region'].astype('category')

# Ensure 'Date' is in datetime format and then extract 'Month'
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data['Month'] = data['Date'].dt.month

data['Month_int'] = data['Month'].apply(lambda x : int(x))

# Create 'Month_name' directly from 'Month'
data['Month_name'] = data['Month_int'].apply(lambda x: calendar.month_abbr[x])


# In[102]:


data_result = data.drop(columns='Month').head(3)

# Display the result
data_result


# In[103]:


import pandas as pd

# Assuming 'data' is your DataFrame
data_stats = data[['Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate']]

# Basic descriptive statistics
desc_stats = data_stats.describe().T

# Calculating additional statistics
desc_stats['variance'] = data_stats.var()
desc_stats['range'] = desc_stats['max'] - desc_stats['min']
desc_stats['IQR'] = desc_stats['75%'] - desc_stats['25%']

# Rounding the result to two decimal places
desc_stats = desc_stats.round(2)

# Display the extended descriptive statistics
desc_stats


# In[104]:


import pandas as pd

# Assuming 'data' is your DataFrame
# Grouping by 'Region' and calculating various statistics
region_stats_extended = data.groupby('Region')[['Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate']].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()

# Flatten the MultiIndex in columns
region_stats_extended.columns = ['_'.join(col).strip() if col[1] else col[0] for col in region_stats_extended.columns.values]

# Rounding the numeric values to two decimal places for better readability
numeric_cols = region_stats_extended.select_dtypes(include=['float64', 'int64']).columns
region_stats_extended[numeric_cols] = region_stats_extended[numeric_cols].round(2)

# Display the extended region statistics
region_stats_extended


# In[105]:


import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
# Selecting the columns for the correlation heatmap
heat_maps = data[['Estimated Unemployment Rate', 'Estimated Employed', 
                'Estimated Labour Participation Rate', 'longitude', 
                'latitude', 'Month_int']]

# Calculating the correlation matrix
correlation_matrix = heat_maps.corr()

# Setting up the matplotlib figure
plt.figure(figsize=(16, 8))
sns.set_context('notebook', font_scale=1.2)

# Creating the heatmap with a different color palette and adjusting annotations
sns.heatmap(correlation_matrix, annot=True, cmap='rainbow', fmt=".3f", linewidths=0.5, linecolor='blue')

# Adding a title for the heatmap
plt.title('Correlation Heatmap of Employment and Geographical Metrics', fontsize=16)

# Adjusting the aspect ratio for better display
plt.gca().set_aspect('equal', adjustable='box')

# Display the heatmap
plt.show()


# In[106]:


import plotly.express as px

# Creating a box plot for the unemployment rate by states
box_plot = px.box(data_frame=data, x='States', y='Estimated Unemployment Rate',
                  color='States', title='Box Plot of Unemployment Rate by State')

# Customizing the layout
box_plot.update_layout(
    xaxis_title="States",
    yaxis_title="Unemployment Rate",
    xaxis={'categoryorder': 'total descending'},
    template='plotly_dark'  # Changing the template for a different style
)

# Adding additional customizations
box_plot.update_traces(marker=dict(size=5),
                       boxmean='sd')  # Show mean and standard deviation

# Showing the plot
box_plot.show()


# In[107]:


import plotly.express as px

# Selecting dimensions for scatter matrix
selected_dimensions = ['Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate']

# Creating the scatter matrix plot
scatter_matrix_plot = px.scatter_matrix(data_frame=data, 
                                        dimensions=selected_dimensions, 
                                        color='Region', 
                                        title='Scatter Matrix of Unemployment, Employment, and Participation Rates by Region',
                                        labels={col: col.replace('_', ' ') for col in selected_dimensions}) # Improve readability of axis labels

# Customizing the template and adding more features for interactivity
scatter_matrix_plot.update_layout(
    template='plotly_white',  # Using a light theme for a change
    height=800,  # Adjusting the plot size
    width=800
)

scatter_matrix_plot.update_traces(diagonal_visible=False,  # Hides the histogram/distribution in the diagonal
                                  marker=dict(size=4, opacity=0.7))  # Adjust marker size and opacity

# Displaying the plot
scatter_matrix_plot.show()


# In[108]:


import matplotlib.pyplot as plt
import seaborn as sns

# Creating a box plot for the unemployment rate by states
plt.figure(figsize=(12, 6))
sns.boxplot(x='States', y='Estimated Unemployment Rate', data=data, palette='rainbow')
plt.title('Box Plot of Unemployment Rate by State')
plt.xticks(rotation=45)
plt.ylabel('Unemployment Rate (%)')
plt.xlabel('States')
plt.grid(True)
plt.show()


# In[109]:


import matplotlib.pyplot as plt

# Scatter plot of Employment vs. Unemployment Rate
plt.figure(figsize=(10, 6))
plt.scatter(data['Estimated Employed'], data['Estimated Unemployment Rate'], c='blue', alpha=0.5)
plt.title('Scatter Plot of Employment vs. Unemployment Rate')
plt.xlabel('Estimated Employed')
plt.ylabel('Estimated Unemployment Rate')
plt.grid(True)
plt.show()


# In[110]:


import matplotlib.pyplot as plt

# Creating a histogram for Labour Participation Rate
plt.figure(figsize=(10, 6))
plt.hist(data['Estimated Labour Participation Rate'], bins=20, color='red', alpha=0.7)
plt.title('Histogram of Labour Participation Rate')
plt.xlabel('Labour Participation Rate')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()


# In[111]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the mean unemployment rate for each state
mean_unemp_by_state = data.groupby('States')['Estimated Unemployment Rate'].mean().reset_index()

# Sorting the data by unemployment rate
sorted_mean_unemp = mean_unemp_by_state.sort_values(by='Estimated Unemployment Rate')

# Creating the bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='States', y='Estimated Unemployment Rate', data=sorted_mean_unemp, palette='tab10')

# Adding titles and labels
plt.title('Average Unemployment Rate by State')
plt.xlabel('States')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(rotation=45)

# Display the plot
plt.show()


# In[112]:


import plotly.express as px

# Load your DataFrame 'data'
# Your existing code for preparing the DataFrame goes here

# Define a custom color palette
custom_colors = ['#FF5733', '#33C1FF', '#FFC733', '#33FF57', '#8C33FF', '#FF33F0', '#33FFF0']

# Alternative approach to create an animated bar plot with custom colors
fig = px.bar(data_frame=data, 
             x='Region', 
             y='Estimated Unemployment Rate', 
             animation_frame='Month_name', 
             color='States',
             height=700,
             title='Monthly Unemployment Rate by Region (2020)',
             color_discrete_sequence=custom_colors)

# Update layout for a better visual presentation
fig.update_layout(xaxis=dict(categoryorder='total descending'), 
                  template='plotly',
                  xaxis_title='Region',
                  yaxis_title='Unemployment Rate (%)')

# Adjust animation speed
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2000

# Show the plot
fig.show()


# In[127]:


import plotly.express as px

# Load your DataFrame 'data'
# Your existing code for preparing the DataFrame goes here

# Create a subset of data focusing on unemployment rates
unemplo_data = data[['States', 'Region', 'Estimated Unemployment Rate']]
unemplo = unemplo_data.groupby(['Region', 'States'])['Estimated Unemployment Rate'].mean().reset_index()

# Use a Treemap to visualize the data
fig = px.treemap(unemplo, 
                 path=['Region', 'States'], 
                 values='Estimated Unemployment Rate',
                 color='Estimated Unemployment Rate',
                 color_continuous_scale='turbo',  # Change to a different color scale
                 title='Average Unemployment Rate by Region and State',
                 height=650,
                 template='seaborn')  # Change to a different template for a new look

# Show the plot
fig.show()


# In[128]:


# Filtering data to separate before and during lockdown periods
before_lockdown = data[data['Month_int'].between(1, 4)]
during_lockdown = data[data['Month_int'].between(4, 7)]

# Grouping by states and calculating mean unemployment rates for each period
mean_unemp_before = before_lockdown.groupby('States')['Estimated Unemployment Rate'].mean().reset_index()
mean_unemp_during = during_lockdown.groupby('States')['Estimated Unemployment Rate'].mean().reset_index()

# Renaming columns for clarity
mean_unemp_before.rename(columns={'Estimated Unemployment Rate': 'Unemp_Rate_Before_Lockdown'}, inplace=True)
mean_unemp_during.rename(columns={'Estimated Unemployment Rate': 'Unemp_Rate_During_Lockdown'}, inplace=True)

# Merging the two DataFrames for a comparative view
comparison_df = mean_unemp_during.merge(mean_unemp_before, on='States')

# Displaying the first two rows of the merged DataFrame
comparison_df.head(2)


# In[129]:


# Calculating the percentage change in unemployment rate
comparison_df['Percentage_Change_Unemployment'] = (
    (comparison_df['Unemp_Rate_During_Lockdown'] - comparison_df['Unemp_Rate_Before_Lockdown']) /
    comparison_df['Unemp_Rate_Before_Lockdown'] * 100
).round(2)

# Sorting data by percentage change
sorted_data = comparison_df.sort_values('Percentage_Change_Unemployment')

# Creating a bar plot for percentage change in unemployment
fig = px.bar(sorted_data, x='States', y='Percentage_Change_Unemployment', 
             color='Percentage_Change_Unemployment',
             title='Percentage Change in Unemployment Rate by State Post-Lockdown',
             template='plotly_dark')

# Display the plot
fig.show()


# In[130]:


import pandas as pd

# Assuming 'data' is your initial DataFrame with unemployment information
# Filter the DataFrame for periods before and during the lockdown
before_lockdown = data[data['Month_int'] <= 3]
during_lockdown = data[data['Month_int'] >= 4]

# Calculate mean unemployment rates for each state before and during the lockdown
mean_unemp_before = before_lockdown.groupby('States')['Estimated Unemployment Rate'].mean().reset_index()
mean_unemp_during = during_lockdown.groupby('States')['Estimated Unemployment Rate'].mean().reset_index()

# Rename columns for clarity
mean_unemp_before.rename(columns={'Estimated Unemployment Rate': 'Unemployment Rate before lockdown'}, inplace=True)
mean_unemp_during.rename(columns={'Estimated Unemployment Rate': 'Unemployment Rate after lockdown'}, inplace=True)

# Merge the two DataFrames for a comparative view
g_lock = mean_unemp_during.merge(mean_unemp_before, on='States')

# Now continue with the percentage change calculation
g_lock['unemployment_percentage_change'] = round(
    (g_lock['Unemployment Rate after lockdown'] / g_lock['Unemployment Rate before lockdown'] - 1) * 100, 2
)

# Sort the DataFrame based on the percentage change
sorted_data = g_lock.sort_values(by='unemployment_percentage_change', ascending=False)
sorted_data

# Proceed with creating the bar plot as previously described


# In[131]:


plot_per = g_lock.sort_values('unemployment_percentage_change')


# In[132]:


import pandas as pd
import plotly.express as px
import numpy as np

# States list
states = ['Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Delhi', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh',
          'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Meghalaya', 'Odisha', 
          'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'West Bengal']

# Generating random unemployment rates for each state
unemployment_rates = np.random.uniform(5, 20, len(states))

# Creating a DataFrame
df = pd.DataFrame({'State': states, 'Unemployment Rate': unemployment_rates})

# Sorting the DataFrame based on Unemployment Rate for better visualization
df_sorted = df.sort_values('Unemployment Rate')

# Creating the bar chart
fig = px.bar(df_sorted, x='Unemployment Rate', y='State', color='Unemployment Rate',
             title='Unemployment Rates Across States in India', orientation='h',
             template='ggplot2', height=650)

# Display the figure
fig.show()


# In[133]:


# Function to categorize the impact based on the percentage change
def categorize_impact(percentage_change):
    if percentage_change <= 10:
        return 'Low Impact'
    elif percentage_change <= 20:
        return 'Moderate Impact'
    elif percentage_change <= 30:
        return 'High Impact'
    elif percentage_change <= 40:
        return 'Very High Impact'
    else:
        return 'Extreme Impact'

# Applying the categorization function to the DataFrame
plot_per['Impact Category'] = plot_per['unemployment_percentage_change'].apply(categorize_impact)

# Creating a horizontal bar chart using Plotly Express
fig = px.bar(plot_per, 
             y='States', 
             x='unemployment_percentage_change', 
             color='Impact Category',
             title='Impact of Lockdown on Employment Across States',
             labels={'percentage change in unemployment': 'Percentage Change in Unemployment Rate', 'States': 'States'},
             orientation='h',  # Horizontal bar chart
             height=650,
             template='plotly_white')

# Customizing the layout for better readability
fig.update_layout(xaxis_title='Percentage Change in Unemployment Rate',
                  yaxis_title='States',
                  xaxis={'categoryorder':'total ascending'},  # Sorting states based on unemployment change
                  coloraxis_colorbar=dict(title='Impact Category'))

# Displaying the figure
fig.show()


# In[146]:


# Select relevant columns for clustering
columns_to_use = ['Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate']
cluster_data = data[columns_to_use]

# Handling missing values (if any)
# Here, we drop rows with missing values, but you can choose other methods like imputation
cluster_data = cluster_data.dropna()

# Standardize the data
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)


# In[147]:


# Elbow method to determine k
inertia = []
for i in range(1, 11):  # testing 1 to 10 clusters
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(cluster_data_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the results
import matplotlib.pyplot as plt

plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[148]:


from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix, accuracy_score


# Print the column names to verify
print("Columns in the dataset:", data.columns.tolist())

# Assuming the correct column names are found, replace them in the following lines
# Replace 'your_unemployment_rate_column' and 'your_labour_participation_rate_column' with actual column names
unemployment_rate_col = 'Estimated Unemployment Rate'
labour_participation_rate_col = 'Estimated Labour Participation Rate'

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
data['Month'] = data['Date'].dt.month

# Selecting relevant features for clustering
features_for_clustering = data[[unemployment_rate_col, labour_participation_rate_col]].dropna()

# Applying k-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features_for_clustering)

# Adding cluster information to the original data
data['Cluster'] = clusters

# Visualizing the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=unemployment_rate_col, y=labour_participation_rate_col, 
                hue='Cluster', data=data, palette='viridis')
plt.title('Clusters based on Unemployment Rate and Labour Participation Rate')
plt.show()

# Calculate silhouette score
silhouette_avg = silhouette_score(features_for_clustering, clusters)

print(f"Silhouette Score: {silhouette_avg:.2f}")


# In[ ]:





# In[ ]:




