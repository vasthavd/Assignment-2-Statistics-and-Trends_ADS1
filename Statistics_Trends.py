# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:22:35 2023

@author: vasth
"""

import pandas as pd

def read_world_bank_csv(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename, header=2)
    df = df.drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)
    #df = df.dropna()
    df2 = df.transpose()
    df2.columns = df2.iloc[0]  # Set the first row as the column names
    df = df.set_index('Country Name')
    #df2 = df2.set_index(('Country Name', ''), append=True)
    #df2 = df2.rename_axis('Year')
    df2 = df2.rename_axis(columns='Country Name', index='Year')
    df = df.rename_axis(index='Country Name', columns='Year')

    df2 = df2.iloc[1:]  # Drop the first row
    return df, df2

data, area = read_world_bank_csv('API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_5348553.csv')
print(data.head())
print(area.head())
