# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:22:35 2023

@author: vasth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cluster_tools as ct
import stats as st

def read_world_bank_csv(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename, header=2)
    df = df.drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)
    df.dropna()
    df.dropna(how='all', axis=1, inplace=True)

    df2 = df.transpose()
    df2.columns = df2.iloc[0]  # Set the first row as the column names
    df = df.set_index('Country Name')
    #df2 = df2.set_index(('Country Name', ''), append=True)
    #df2 = df2.rename_axis('Year')
    df2 = df2.rename_axis(columns='Country Name', index='Year')
    df = df.rename_axis(index='Country Name', columns='Year')

    df2 = df2.iloc[1:]  # Drop the first row
    return df, df2

data, area = read_world_bank_csv('Electricity.csv')
print(data.head())
print(area.head())

# Percentage of electricity
# Select the 10 countries to plot
countries = ['United States', 'China', 'Japan', 'Germany',  'India', 
             'United Kingdom', 'France', 'Brazil', 'Italy', 'Canada']

# Get the data for these countries
data_countries = data.loc[countries]

# Plot the data
plt.figure(figsize=(10, 5)) # Set the size of the figure
for country in countries:
    plt.plot(data.columns, data_countries.loc[country], label=country)
plt.xlabel('Year') # Set the label for the x-axis
plt.xticks(rotation=90)
plt.ylabel('% of population with access to electricity') # Set the label for the y-axis
plt.title('Access to Electricity by Country') # Set the title for the plot
plt.legend() # Show the legend

#urban population

up, up_t = read_world_bank_csv('urbanp.csv')


# Get the data for these countries
up_countries = up.loc[countries]

# Plot the data
plt.figure(figsize=(7, 5)) # Set the size of the figure
for country in countries:
    plt.plot(up.columns, up_countries.loc[country], label=country)
plt.xlabel('Year') # Set the label for the x-axis
plt.xlim(min(up.columns), max(up.columns))
plt.ylabel('Urban population') # Set the label for the y-axis
plt.xticks(np.arange(0, len(up.columns)+1, 5), rotation = 90)
#plt.xticks(rotation = 90)
plt.title('Population') # Set the title for the plot

plt.legend() # Show the legend
#plt.xticks(np.arange(up.columns,10)) 


#methane

methane, methane_t = read_world_bank_csv('Methane.csv')


# Extract the data for the 5 countries
#countries = ['China', 'United States', 'India', 'Canada', 'Japan']
years = ['2015', '2016', '2017', '2018', '2019']
usage = up.loc[countries, years]

# Plot the bar plot
usage.plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Electricity usage')
plt.title('Electricity usage in 5 countries from 2015 to 2019')


print(methane.describe())
print(usage)


fd = pd.merge(data['2015'], up['2015'], left_index=True, right_index=True)
fd = pd.merge(fd, methane['2015'], left_index=True, right_index=True)
fd.columns = ['Electricity', 'Urbanp', 'Methane']
# Show the merged dataframe
print(fd)
plt.figure()
corr_matrix = fd.corr(method='pearson')

# Plot the correlation heatmap using Seaborn
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')



plt.show()

