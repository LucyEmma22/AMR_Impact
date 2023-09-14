#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import statsmodels.api as sm
from scipy.stats.stats import spearmanr
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics

##############################################################################################################################

def normalise_by_age(data, response, min_year, max_year):
    data = data[data['year'] <= max_year] # Remove papers newer than max_year
    data = data[data['year'] >= min_year] # Remove papers older than min_year

    # Plot Histogram of Age
    plt.hist(data['age'].tolist(), bins=20)
    plt.xlabel('Age (Years)')
    plt.ylabel("Frequency")
    plt.show()

    # Plot Histogram of Response
    plt.hist(data[response].tolist(), bins=20)
    plt.xlabel(response)
    plt.ylabel("Frequency")
    plt.show()

    # Spline model
    knots = np.linspace(data['age'].min(), data['age'].max(), num=8)[1:-1] # Define list of knots for the spline model
    spline_model = sm.GLM.from_formula(f'{response} ~ bs(age, knots = knots)', data=data).fit() # Spline model using age of paper to predict Altmetric Score
    preds = spline_model.get_prediction().summary_frame(alpha=0.05)['mean'] # Spline model output
    data = data.merge(preds, left_index=True, right_index=True) # Join spline model results with data
    data['residual'] = data[response] - data['mean'] # Calculate each papers residual from the spline model 
    line_data = data[['age','mean']].drop_duplicates().sort_values(by='age') # Select columns and drop duplicates for plotting the line

    plt.hist2d(data['age'], data[response], bins=50, cmap='viridis',norm=colors.LogNorm())
    plt.colorbar(label='Density')
    plt.plot(line_data['age'], line_data['mean'], linewidth=1, c="black")
    plt.xlabel('Age (Years)')
    plt.ylabel(response)
    plt.show()

    # Linear model
    model = np.polyfit(data['age'], data[response], 1)
    model_predict = np.polyval(model, data['age'])
    plt.hist2d(data['age'], data[response], bins=50, cmap='viridis',norm=colors.LogNorm())
    plt.colorbar(label='Density')
    plt.plot(data['age'], model_predict, c='black')
    plt.xlabel('Age (Years)')
    plt.ylabel(response)
    plt.show()
    return data


# Define papers as high or low impact depending on various thresholds 
def define_impact(data):
    data = data.sort_values(by ='residual', ascending=False) # Sort by residual
    data['order'] = list(range(0, len(data))) # add a column called orde
    impact_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] # define proportion of papers to be high impact
    for threshold in impact_levels:
        data[f'impact_{threshold}'] = np.where(data['order'] < (round(len(data)*threshold)), 1, 0)
    data.drop(['order'], axis=1, inplace=True)
    data.sort_index(inplace=True)
    return data


# Plot Cumultive Distribution and Calculate Gini Coefficient
def plot_cumulative_distribution(data, response):
    data = data.sort_values(by ='residual', ascending=False) # Sort by residual
    data['cumulative'] = data[response].cumsum()
    impact_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] # define proportion of papers to be high impact
    plt.plot(np.array(data['cumulative']))
    x_values = [round(len(data['cumulative'])*i) for i in impact_levels]
    y_values = [data['cumulative'].iloc[x] for x in x_values]
    for i in range(len(x_values)):
        x = x_values[i]
        y = y_values[i]
        plt.plot([x, x], [0, y], color='red', linestyle='--')
        plt.plot([0, x], [y, y], color='red', linestyle='--')
    plt.xlabel("Number of Abstracts")
    plt.ylabel(response)
    plt.show()
    for impact,y in zip(impact_levels,y_values):
        print (str(round(impact*100)) + "% high impact = ", str(np.round(y/max(data['cumulative'])*100,1)) + "% of citations")
        
    array = np.array(data[response]).flatten()
    if np.amin(array) < 0:
        array -= np.amin(array) # Values cannot be negative:
    array = array + 0.0000001 # Values cannot be 0
    array = np.sort(array) # Values must be sorted
    index = np.arange(1,array.shape[0]+1) # Index per array element
    n = array.shape[0] # Number of array elements
    print ('Gini Coefficient: ' + str(round((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)),5))) # Gini coefficient
    
    data.drop(['cumulative'], axis=1, inplace=True)
    data.sort_index(inplace=True)

##############################################################################################################################

# Import Text
preprocessed_text = pd.read_csv("preprocessed_text_filtered.csv").dropna(subset='abstract') 
preprocessed_text['age'] = (2023 + 9/12) - preprocessed_text['year'] + preprocessed_text['month']/12 # Calculate age (from Sept 2023)

# Normalise Citations
citation_data = pd.read_csv('citation_data.csv').merge(preprocessed_text, on='PMID', how='inner')
citation_data['log_cpy'] = np.log((citation_data['number_citations'] / citation_data['age'])+1) # Calculate citations per year
citation_data = normalise_by_age(citation_data, 'log_cpy', 1970, 2021)
citation_data = define_impact(citation_data)
plot_cumulative_distribution(citation_data, 'number_citations')
citation_data.to_csv("preprocessed_normalised_citations.csv", index=False)

# Normalise Altmetrics
altmetric_data = pd.read_csv('altmetric_data.csv')
mention_columns = [col for col in altmetric_data.columns if col.endswith('mentions')]
altmetric_data = altmetric_data[['PubMed ID', 'Altmetric Attention Score', 'Number of Mendeley readers', 'Number of Dimensions citations'] + mention_columns].rename(columns={'PubMed ID': 'PMID', 'Altmetric Attention Score':'altmetric_score'}).merge(preprocessed_text, on='PMID', how='inner')
for x in ['Policy mentions', 'Patent mentions']:
    altmetric_data.loc[altmetric_data[x] != 0, x] = 1
altmetric_data = altmetric_data.rename(columns={'Policy mentions':'policy_mentions', 'Patent mentions':'patent_mentions'})
altmetric_data['log_altmetric_score'] = np.log(altmetric_data['altmetric_score']+1) # Rescale altmetric score
altmetric_data = normalise_by_age(altmetric_data, 'log_altmetric_score', 1970, 2024)
altmetric_data = define_impact(altmetric_data)
plot_cumulative_distribution(altmetric_data, 'altmetric_score')
altmetric_data.to_csv("preprocessed_normalised_altmetrics.csv", index=False)

##############################################################################################################################

# Showing that citation rate and altmetric score are highly correlated

def plot_relationship (data, x, y):
    model = np.polyfit(data[x], data[y], 1)
    model_predict = np.polyval(model, data[x])
    plt.hist2d(data[x], data[y], bins=100, cmap='viridis',norm=colors.LogNorm())
    plt.colorbar(label='Density')
    plt.plot(data[x], model_predict, c='black')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    print('Linear Model Slope: ' , str(np.round(model[0],3)))
    print('Spearman Correlation: ', str(np.round(spearmanr(data[x], data[y]),3)))
    
def plot_confusion_matrix(data, x, y):
    cm = confusion_matrix(data[y], data[x], labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    print('Accuracy = %.3f' % metrics.accuracy_score(data[y], data[x]))
    print('Classification report:')
    print(metrics.classification_report(data[y], data[x]))

data = citation_data[['PMID', 'residual']].rename({'residual': 'citation_rate'}, axis=1).merge(altmetric_data[['PMID', 'residual', 'policy_mentions', 'patent_mentions']].rename({'residual': 'altmetric_score'}, axis=1),on='PMID', how='inner')
plot_relationship(data, 'citation_rate', 'altmetric_score')
plot_confusion_matrix(data, 'policy_mentions','patent_mentions')

for x in ['policy_mentions', 'patent_mentions']:
    data.loc[data[x] != 0, x] = 'Yes'
    data.loc[data[x] == 0, x] = 'No'
sns.boxplot(data=data, x="citation_rate", y="policy_mentions",
            showcaps=False,
            flierprops={"marker": "none"})
sns.boxplot(data=data, x="citation_rate", y="patent_mentions",
            showcaps=False,
            flierprops={"marker": "none"})