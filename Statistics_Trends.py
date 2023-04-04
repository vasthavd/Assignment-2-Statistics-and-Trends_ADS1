# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:22:35 2023

@author: vasth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_world_bank_csv(file_name):
    
    """
    This function reads a CSV file from the World Bank data
    and returns two DataFrames.

    Parameters
    ----------
    file_name : str
    The path to the CSV file.

    Returns
    -------
    year_as_column : DataFrame
    The DataFrame containing the data from the CSV file with years as columns.

    country_as_column : DataFrame
    The DataFrame containing the data from the CSV file, transposed with 
    countries as columns.

    Raises
    ------
    FileNotFoundError
    If the CSV file does not exist.

    IOError
    If there is an error reading the CSV file.

    """
    # Reading the CSV file into a DataFrame
    year_as_column = pd.read_csv(file_name, header=2)

    # Dropping the first three columns, which are the header, 
    # Indicator Code, and Country Code.
    year_as_column = year_as_column.drop(['Indicator Code', 'Country Code',
                                          'Indicator Name'], axis=1)

    # Dropping any rows that contain missing values and cleaning dataframe.
    year_as_column.dropna()
    year_as_column.dropna(how='all', axis=1, inplace=True)
    
    # Setting the index of the DataFrame to the Country Name column.
    year_as_column = year_as_column.set_index('Country Name')

    # Renaming the axis of the DataFrame to Years.
    year_as_column = year_as_column.rename_axis(index='Country Name',
                                                columns='Year')

    # Transposing the DataFrame.
    country_as_column = year_as_column.transpose()

    # Renaming the axis of the DataFrame to Countries.
    country_as_column = country_as_column.rename_axis(columns='Country Name', 
                                                      index='Year')

    # Dropping the first row of the DataFrame, which is just the column names.
    country_as_column = country_as_column.iloc[1:]

    # Returns the two DataFrames.
    return year_as_column, country_as_column


def group_years_by_count(year_as_column_df):
    
    """
    Group years based on non-null count.

    This function takes a pandas dataframe as input, and creates a new    
    dataframe with non-null count and year columns. The years are then grouped
    into two categories based on whether their non-null count is above or below 
    the median count. The resulting groups are printed with a message 
    indicating the count, which is useful to determine years selection for 
    analysis.

    Parameters
    ----------
    year_as_column_df : pandas DataFrame
    A dataframe with one column containing year data.

    Returns
    -------
    result: str
    A string with two sections separated by a blank line. The first section
    lists the years with non-null count above the median count, and the
    second section lists the years with non-null count less than or equal
    to the median count.
    
    """
    
    # Create dataframe with non-null count and year columns
    non_null_years = year_as_column_df.notnull().sum()\
        .sort_values(ascending = False).to_frame().reset_index()
    non_null_years.columns = ['Year', 'Non-null Count']

    # Calculate median non-null count
    median_count = non_null_years['Non-null Count'].median()

    # Group years based on non-null count above or below median
    non_null_groups = non_null_years.groupby(
        non_null_years['Non-null Count'] > median_count)\
        .apply(lambda x: x['Year'].tolist())

    # Print groups with messages indicating count
    result = []
    result.append(
        f"Years with non-null count above the median count ({median_count}):")
    result.append(str(non_null_groups[True]))
    result.append("")
    result.append(
        f"Years with non-null count <= median count ({median_count}):")
    result.append(str(non_null_groups[False]))
    return '\n'.join(result)


def line_plot(year_as_column_df, xlabel, ylabel, title):
    
  """
  Plots a line plot of the given DataFrame.

  Parameters
  ----------
    year_as_column_df: pandas DataFrame
    The DataFrame to plot.
    xlabel: str
    The label for the x-axis.
    ylabel: str
    The label for the y-axis.
    title: str
    The title for the plot.

  Returns
  -------
  None, Plots a matplotlib figure object: Line - Plot.
  
  """
  # Select the 10 countries to plot in order of the GDP list.
  countries_list = ['United States', 'China', 'Japan', 'Germany',  'India', 
                    'United Kingdom', 'France', 'Brazil', 'Italy', 'Canada']

  # Get the data for these countries.
  sorted_countries_list = year_as_column_df.loc[countries_list]

  # Plot the data.
  plt.figure()
  for country in countries_list:
    plt.plot(year_as_column_df.columns, 
             sorted_countries_list.loc[country], label=country)
    
  # Set the x-axis limits to the minimum and maximum years in the DataFrame.
  plt.xlim(min(year_as_column_df.columns), max(year_as_column_df.columns))
  
  # Set the x-axis ticks to the years, spaced every 5 years.
  plt.xticks(np.arange(0, len(year_as_column_df.columns)+1, 5), rotation = 90)

  # Set the label for the x-axis.    
  plt.xlabel(xlabel)
  
  # Set the label for the y-axis.
  plt.ylabel(ylabel)
  
  # Set the title for the plot.
  plt.title(title)
  
  # Add a legend to the plot, being labeled with the corresponding country.
  plt.legend(bbox_to_anchor=(1.05, 1))
  
  # Return the current figure object.
  return plt.gcf()


def bar_plot(year_as_column_df, xlabel, ylabel, title):
    
  """
  Plots a bar plot of the given DataFrame.

  Parameters
  ----------
   year_as_column_df: pandas DataFrame
   The DataFrame to plot.
   xlabel: str
   The label for the x-axis.
   ylabel: str
   The label for the y-axis.
   title: str
   The title for the plot.
    
  Returns
  -------
      None, Plots a matplotlib figure object : Bar - Plot.
  """
  # Select the 10 countries to plot in order of the GDP list.
  countries_list = ['United States', 'China', 'Japan', 'Germany',  'India', 
                    'United Kingdom', 'France', 'Brazil', 'Italy', 'Canada']
  # Select the 5 years to plot which have large nuber of entries.
  years = ['2015', '2016', '2017', '2018', '2019']
  # Get the data for these countries.
  sorted_countries_list = year_as_column_df.loc[countries_list, years]
  
  
  # Plot the data.
  plt.figure()
  sorted_countries_list.plot(kind='bar')

  # Set the label for the x-axis.
  plt.xlabel(xlabel)

  # Set the label for the y-axis.
  plt.ylabel(ylabel)

  # Set the title for the plot.
  plt.title(title)
  
  # Add a legend to the plot, being labeled with the corresponding year.
  plt.legend(loc='upper right')
  
  # Return the current figure object.
  return plt.gcf()


def correlation_heatmap(year):
    
    """
    This function creates a correlation heatmap for a given year by merging 
    data from various sources and calculating a correlation matrix 
    between the variables.

    Parameters
    ----------
    year : int 
    The year for which the correlation heatmap needs to be created.

    Returns
    -------
    None , Plots a correlation heatmap.
    
    """
    
    # Merge the data for the given year
    m_df = pd.merge(ate[year], urban_pop[year],
                  left_index=True, right_index=True)
    m_df = pd.merge(m_df, methane[year], left_index=True, right_index=True)
    m_df = pd.merge(m_df, forest_area[year], left_index=True, right_index=True)
    m_df = pd.merge(m_df, gdp[year], left_index=True, right_index=True)
    m_df = pd.merge(m_df, agri[year], left_index=True, right_index=True)
    m_df = pd.merge(m_df, ce[year], left_index=True, right_index=True)
    m_df = pd.merge(m_df, energy_use[year], left_index=True, right_index=True)
    m_df = pd.merge(m_df, gpi[year], left_index=True, right_index=True)
    m_df.columns = ['Electricity', 'Urban_pop', 'Methane', 'Forest',
                  'GDP', 'Agri land', 'C02', 'Energy_Use', 'GPI']

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Calculate the correlation matrix
    corr_matrix = m_df.corr()

    # Create a heatmap
    heat_map = ax.imshow(corr_matrix, cmap='coolwarm')

    # Set the ticks and tick labels
    ax.set_xticks(range(len(m_df.columns)))
    ax.set_yticks(range(len(m_df.columns)))
    ax.set_xticklabels(m_df.columns)
    ax.set_yticklabels(m_df.columns)

    # Rotate the tick labels and set them at the center
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')

    # Add the correlation values inside the heatmap
    for i in range(len(m_df.columns)):
        for j in range(len(m_df.columns)):
            text = ax.text(j, i, round(corr_matrix.iloc[i, j], 2),
                           ha="center", va="center", color="w")

    # Add a colorbar
    cbar = ax.figure.colorbar(heat_map, ax=ax)

    # Set the title and show the plot
    ax.set_title(f"Correlation Heatmap for {year}")
    plt.tight_layout()

    
# Indicator : Access to Electricity(% of Population)

ate, ate_t = read_world_bank_csv('Access to Electricity(% of Population).csv')
print('\nAccess to Electricity(% of Population)')
print(ate.head())
print(ate_t.head())
print(ate.describe())
print(group_years_by_count(ate))

line_plot(ate, 'Year', '% of population', 
                            'Access to Electricity for GDP list countries')

# Indicator : Urban population

urban_pop, urban_pop_t = read_world_bank_csv('Urban population.csv')
print('\nUrban population')
print(urban_pop.describe())

line_plot(urban_pop, 'Year', 
          'Urban populatio in 10 Millions', 
          'Urban population growth for GDP list countries')

# Indicator : Methane emissions (kt of CO2 equivalent)

methane, methane_t = read_world_bank_csv(
    'Methane emissions (kt of CO2 equivalent).csv')
print('\nMethane emissions (kt of CO2 equivalent)')
print(methane.describe())
bar_plot(methane, 'Country', 
         'Emissions in KT', 'Methane Emissions for GDP list countries')


# Indicator name: Forest area (% of land area)

forest_area, forestarea_t = read_world_bank_csv(
    'Forest area (% of land area).csv')
print('\nForest area (% of land area)')
print(group_years_by_count(forest_area))
print(forest_area.describe())

bar_plot(forest_area, 'Country', 'Forest area (% of land area)', 
         'Forest usage for GDP list countries from 2015 to 2019')


# Indicator name: GDP growth(annual %)

gdp, gdp_t = read_world_bank_csv('GDP growth(annual %).csv')
print('\nGDP growth(annual %)')
print(group_years_by_count(gdp))
print(gdp.describe())

bar_plot(gdp, 'Country', 'GDP growth(annual %)', 'GDP growth for 5 years')


#Indicator name: CO2 emmisons(kt)

ce, ce_t = read_world_bank_csv('CO2 emmisons(kt).csv')
print('\nCO2 emmisons(kt)')
print(ce.describe())
bar_plot(ce, 'Country', 'CO2 emmisons(kt)', 
         'CO2 emmisons(kt) growth for GDP list countries over 5 years')


#Indiacator Name: Agricultural land (% of land area)

agri, agri_t = read_world_bank_csv('Agricultural land (% of land area).csv')
print('\nAgricultural land (% of land area)')
print(group_years_by_count(agri))
print(agri.describe())
bar_plot(agri, 'Country', 'Agricultural land (% of land area)', 
         'Agricultural land for GDP list countries over 5 years')


#Indiacator Name: Electric power consumption (kWh per capita)

ec, ec_t = read_world_bank_csv('EPC(kWh per capita).csv')
print('\nElectric power consumption (kWh per capita)')
print(group_years_by_count(ec))
print(ec.describe())

line_plot(ec, 'Year', 'kWh per capita', 
          'Electric power consumption for GDP list countries')

# Indicator Name: School enrollment, primary and secondary (gross), (GPI)

gpi, gpi_t = read_world_bank_csv('GPI.csv')
print('\nSchool enrollment, primary and secondary (gross),(GPI)')
print(group_years_by_count(gpi))
print(gpi.describe())
bar_plot(gpi, 'Country', 'Gender Parity Index', 
         'Gender Parity Index growth for GDP list countries over 5 years')


# Indicator name : Energy use (kg of oil equivalent per capita)

energy_use, energy_t = read_world_bank_csv(
    'Energy use (kg of oil equivalent per capita).csv')
print('\nEnergy use (kg of oil equivalent per capita)')
print(energy_use.describe())
print(group_years_by_count(energy_use))
line_plot(energy_use, 'Year', 'kg of oil equivalent per capita', 
          'Energy use for GDP list countries')


# Correlation Analysis for several years


correlation_heatmap('1990')
correlation_heatmap('2000')
correlation_heatmap('2010')

plt.show()
