**Unemployment rate in India during Covid 2019**

**Abstract** 

Another factor that has made the issue more serious is the COVID-19 epidemic. Early in January 2020, the world saw the global spread of a novel coronavirus strain (SARS-CoV-2) that causes the respiratory illness COVID-19. The World Health Organization issued a worldwide health emergency declaration on January 30, 2020, and a pandemic was verified on March 11. 81% of the world's workforce has been impacted by the shutdown of enterprises to prevent travel and the virus's spread (ILO, 2020a). Individuals are worried about their families' overall financial, social, and economic well-being. 

This analysis's main goal is to evaluate how COVID-19 has affected the labor market in different Indian states. In particular aiming to:

**Objective** 

**Assess the Effect on the Labor Market:**

Examine how unemployment rates have changed in several states to see how the COVID-19 epidemic has impacted the labor market. This involves assessing the degree to which the lockdown periods have affected employment both during and after.

**Determine Differences Between States:**

To determine which states have fared better and which have seen more severe effects, compare the unemployment rates in the various states. This comparison will provide light on how state-level labor markets have fared in relation to one another during the epidemic.

**Classify States according to Effect:**

Sort the states according to the level of influence on respective labor markets. Understanding the range of impacts, from states that have survived comparatively unaffected to those that have been severely affected, will be made easier with the aid of this classification.


Show Temporal and Geographic Trends Visually:

To give a clear image of how the influence has changed over time and in different areas, use spatial and temporal representations. This will make it easier to spot regional trends and patterns in the effect on employment.

**Dataset**

●	States = states.
●	Date = the day on which the rate of unemployment was noted
●	Frequency = Monthly Frequency Measurement
●	The percentage of jobless persons in each Indian state is known as the estimated unemployment rate (%).
●	Estimated Employed = Total number of workers
●	Estimated Labor Participation Rate (%) = This is the percentage of the working population in the 16–64 age range that are either employed or looking for work in the economy.

**Preparing Data Model Evaluation **

Initial Exploration and Data Loading:

●	The first thing your script does is load data from a Google Drive-hosted CSV file.
●	Each column in your DataFrame has relevant information printed out about it by the dataset_info function.
●	The data has a section where you may verify any missing values.

Preparing data:

●	You've mentioned managing date formats, changing data types, and renaming columns.
●	The method used to transform "Region" and "Frequency" into categorical categories is suitable.
●	For temporal analysis, obtaining the month and year from the 'Date' column is helpful.

Analytical Statistics:

●	For some columns, descriptive statistics are calculated.
●	The process of computing various statistics and grouping data by area is done properly.
●	An extended correlation heatmap, which is helpful for determining correlations between variables, is created by your script.

Information Visualization:

●	Matplotlib and Plotly are used to construct a variety of visualizations, including box plots, scatter plots, histograms, and bar charts.
●	Your study gains dynamism when you utilize Plotly for interactive graphs (such as geospatial plots and treemaps).
●	Adding distinct color schemes and template customizations to plots improves their visual attractiveness.

Classifying Data:

●	Developing a function to classify states according to the extent of their influence on unemployment.
●	putting the DataFrame through this classification and displaying the outcomes as a bar chart.

Grouping Using k-Means:

●	The k-Means clustering technique and data standardization are included in the script.
●	It is a good idea to use the Elbow approach to figure out the ideal number of clusters.
●	Understanding the grouping is aided by seeing the clusters.

The script is extensive, addressing every facet of data analysis, from sophisticated visualization and clustering to fundamental investigation. It offers a thorough analysis of unemployment statistics and ideas that may be useful for comprehending economic patterns, particularly in light of the epidemic. In order to meet various analytical demands, a multifaceted knowledge of the data is ensured through the use of several display methods.


**Results**

Data pretreatment techniques like datetime conversion and feature extraction are followed before using k-means clustering to unemployment and labor participation statistics. The k-means technique is used, using three clusters, to cluster two important features: the "Estimated Unemployment Rate" and the "Estimated Labour Participation Rate." The outcomes are included back into the dataset, adding dynamism of the labor market to the analysis. The data points that have been clustered according to labor participation and unemployment rates are clearly displayed in the final visualization, a scatter plot, which offers a clear visual depiction of the various labor market divisions.

![image](https://github.com/user-attachments/assets/2832a16e-88a2-4a5e-a9e4-1ee02b4cf6ec)

![image](https://github.com/user-attachments/assets/eba91e05-f5a5-4268-8621-c70fcdbdf4c3)


![image](https://github.com/user-attachments/assets/98c4e4d6-d6bd-4f58-8215-2d772481e83b)
