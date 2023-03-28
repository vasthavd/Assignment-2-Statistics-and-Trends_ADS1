# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:22:35 2023

@author: vasth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# Select the 5 countries to plot
countries = ['United States', 'China', 'India', 'Japan']

# Get the data for these countries
data_countries = data.loc[countries]

# Plot the data
plt.figure(figsize=(10, 6)) # Set the size of the figure
for country in countries:
    plt.plot(data.columns, data_countries.loc[country], label=country)
plt.xlabel('Year') # Set the label for the x-axis
plt.xticks(rotation=90)
plt.locator_params(axis='x', nbins=4)
plt.ylabel('% of population with access to electricity') # Set the label for the y-axis
plt.title('Access to Electricity by Country') # Set the title for the plot
plt.legend() # Show the legend



up, up_t = read_world_bank_csv('urbanp.csv')


# Get the data for these countries
up_countries = up.loc[countries]

# Plot the data
plt.figure(figsize=(10, 6)) # Set the size of the figure
for country in countries:
    plt.plot(up.columns, up_countries.loc[country], label=country)
plt.xlabel('Year') # Set the label for the x-axis
#plt.xlim('1960', '2021')
plt.ylabel('Urban population') # Set the label for the y-axis
plt.xticks(np.arange(0, len(up.columns)+1, 5), rotation = 90)
#plt.xticks(rotation = 90)
plt.title('Population') # Set the title for the plot

plt.legend() # Show the legend
#plt.xticks(np.arange(up.columns,10)) 
plt.show()
