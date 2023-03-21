# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:22:35 2023

@author: vasth
"""

import pandas as pd

def read_world_bank_csv(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_excel(filename, header=3)
    df = df.drop(['Indicator Code', 'Country Code'], axis=1)
    df = df.dropna()
    df2 = df.transpose()
    df_transposed = df2.iloc[3:]
    #df = df.set_index('Country Name')#, 'Indicator Name')

    return df_transposed, df2



data, area = read_world_bank_csv('API_19_DS2_en_excel_v2_4903056.xls')
print(data)

"""
def read_world_bank_csv(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_excel(filename, header=3)
    df = df.drop(['Indicator Code', 'Country Code'], axis=1)
    df = df.set_index('Country Name', 'Indicator Name')
    
    # Extract the country names and set them as the index
    country_names = df.iloc[:, 4:-1]
    df.set_index(country_names, inplace=True)
    df.drop(df.columns[0], axis=1, inplace=True)
    
    # Transpose the DataFrame to make years as columns
    #df1 = df.transpose()
    
    # Reset the index and rename the columns for the second DataFrame
    df2 = df.reset_index()
    df2.rename(columns={'index': 'Year'}, inplace=True)
    df2.set_index('Year', inplace=True)
    
"""