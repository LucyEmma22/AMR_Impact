#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import statsmodels.api as sm
from scipy.stats.stats import spearmanr
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
import ast
import os
os.chdir("/Users/s1995754/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD Year 3/AMR Impact") # set working directory

##############################################################################################################################
bins = 50

def normalise_by_age(data, response, ylab):
    # Spline model
    df = data[['PMID','age', response]].dropna()
    knots = np.linspace(df['age'].min(), df['age'].max(), num=8)[1:-1] # Define list of knots for the spline model
    spline_model = sm.GLM.from_formula(f'{response} ~ bs(age, knots = knots)', data=df).fit() # Spline model using age of paper to predict Altmetric Score
    preds = spline_model.get_prediction().summary_frame(alpha=0.05)['mean'] # Spline model output
    df = df.merge(preds, left_index=True, right_index=True) # Join spline model results with data
    df[f'{response}_residual'] = df[response] - df['mean'] # Calculate each papers residual from the spline model 
    line_data = df[['age','mean']].drop_duplicates().sort_values(by='age') # Select columns and drop duplicates for plotting the line
    data = data.merge(df[['PMID',f'{response}_residual']], on='PMID', how='outer') # Add normalised score to original data

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    fontsize = 16
    
    plt.hist2d(df['age'], df[response], bins=bins, cmap='viridis',norm=colors.LogNorm())
    plt.plot(line_data['age'], line_data['mean'], c='red', linewidth = '3')
    plt.xlabel('Age (Years)', fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.colorbar(label='Density')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(f'Figures/{response}_vs_age.pdf', format='pdf')
    plt.show()
    return data


# Plot Cumultive Distribution and Calculate Gini Coefficient
def plot_cumulative_distribution(data, response, ylab):
    df = data[['PMID','age', response]].dropna()
    df = df.sort_values(by=response, ascending=False) # Sort by residual
    df['cumulative'] = df[response].cumsum()
    
    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    fontsize = 16

    plt.plot(np.array(df['cumulative']))
    plt.xlabel('Number of Abstracts', fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True, useOffset=False))
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True, useOffset=False))
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(4,4))
    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(f'Figures/cumulative_{response}.pdf', format='pdf')
    plt.show() 
    
    # Calculate Gini Index
    array = np.array(df[response]).flatten()
    array = array + 0.0000001 # Values cannot be 0
    array = np.sort(array) # Values must be sorted
    index = np.arange(1,array.shape[0]+1) # Index per array element
    n = array.shape[0] # Number of array elements
    gini = (np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)) 
    return gini


def plot_relationship (data, x, y, xlab, ylab):
    fig, ax = plt.subplots(figsize=(6, 5))
    fontsize = 16

    df = data[[x, y]].dropna()
    model = np.polyfit(df[x], df[y], 1)
    model_predict = np.polyval(model, df[x])
    r_squared = r2_score(df[y], model_predict)
    plt.hist2d(df[x], df[y], bins=bins, cmap='viridis',norm=colors.LogNorm())
    plt.plot(df[x], model_predict, c='red', linewidth = '3')
    plt.xlabel(xlab,fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.colorbar(label='Density')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(f'Figures/{x}_{y}_correlation.pdf', format='pdf')
    plt.show()
    
    print('Linear Model Slope: ' , str(model[0]))
    print('Spearman Correlation: ', str(spearmanr(df[x], df[y])))
    print('R-squared: ', str((r_squared)))
    
    
def plot_distribution(variable, name):
    variable = variable.dropna()
    
    fig, ax = plt.subplots(figsize=(5, 5))
    fontsize = 16
    
    kde = gaussian_kde(variable, bw_method=0.2)
    x = np.linspace(min(variable), max(variable), 100)
    y = kde(x)
    plt.plot(x, y, color='blue', label='KDE')
    plt.fill_between(x, 0, y, alpha=0.3, color='blue')
    plt.xlabel(name, fontsize=fontsize)
    plt.ylabel('Density', fontsize=fontsize)
    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(f'Figures/{name}_distribution.pdf', format='pdf')
    plt.show()
 
    
def plot_frequency (variable, name):
    df = pd.DataFrame(data[variable].explode().value_counts()).reset_index()
    
    fig, ax = plt.subplots(figsize=(5, 5))
    fontsize = 14
    
    plt.bar(df['index'], np.log(df[variable]), color ='blue', width = 0.4)
    plt.xlabel(name, fontsize=fontsize)
    plt.ylabel('Log (Frequency)',fontsize=fontsize)
    plt.xticks(rotation = 90)
    
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(f'Figures/{name}_frequency.pdf', format='pdf')
    plt.show()   
 
##############################################################################################################################

preprocessed_abstracts = pd.read_csv("2_Data_Processing/preprocessed_abstracts2.csv")
citation_data = pd.read_csv('1_Data_Retrieval/citation_data2.csv')
altmetric_data = pd.read_csv('1_Data_Retrieval/altmetric_data2.csv')[['PubMed ID', 'Altmetric Attention Score', 'Policy mentions', 'Patent mentions']].rename(columns={'PubMed ID': 'PMID', 'Altmetric Attention Score':'altmetric_score', 'Policy mentions':'policy_mentions', 'Patent mentions':'patent_mentions'}).drop_duplicates(subset='PMID')
data = preprocessed_abstracts.merge(citation_data, on='PMID', how='left').merge(altmetric_data, on='PMID', how='left')
data['article_type'] = data['article_type'] .apply(lambda x: ast.literal_eval(x))
data = data[data['year'] <= 2022] # Remove papers newer than ...
data = data[data['year'] >= 1970] # Remove papers older than ...
data['age'] = (2023 + 9/12) - data['year'] + data['month']/12 # Calculate age (from Sept 2023) 

# Counts
data['log_cpy'] = np.log(data['number_citations']+1) # Rescale citations
data['log_alt'] = np.log(data['altmetric_score']+1) # Rescale altmetric score
data['log_policy_mentions'] = np.log(data['policy_mentions']+1) # Rescale 
data['log_patent_mentions'] = np.log(data['patent_mentions']+1) # Rescale

##############################################################################################################################
# Cumulative Distributions and Gini Coefficient

plot_cumulative_distribution(data, 'number_citations', 'Cumulative Citations') # Plot cumulative distribution of citations and calculate gini coefficient
plot_cumulative_distribution(data, 'altmetric_score', 'Cumulative Altmetric Score') # Plot cumulative distribution of altmetric score and calculate gini coefficient 
plot_cumulative_distribution(data, 'policy_mentions', 'Cumulative Policy Mentions') # Plot cumulative distribution of altmetric score and calculate gini coefficient 
plot_cumulative_distribution(data, 'patent_mentions', 'Cumulative Patent Mentions') # Plot cumulative distribution of altmetric score and calculate gini coefficient 

# Means
print(np.mean(data['number_citations']))
print(np.mean(data['altmetric_score']))
print(np.mean(data['patent_mentions']))
print(np.mean(data['policy_mentions']))
(data['number_citations'] == 0).sum()
(data['altmetric_score'] == 0).sum()
(data['patent_mentions'] == 0).sum()
(data['policy_mentions'] == 0).sum()
data['number_citations'].count()
data['altmetric_score'].count()
data['patent_mentions'].count()
data['policy_mentions'].count()
(data['number_citations'] == 0).sum()/data['number_citations'].count()
(data['altmetric_score'] == 0).sum()/data['altmetric_score'].count()
(data['patent_mentions'] == 0).sum()/data['patent_mentions'].count()
(data['policy_mentions'] == 0).sum()/data['policy_mentions'].count()


print(np.mean(data['word_count']))
print(np.mean(data['title_word_count']))
print(np.mean(data['number_authors']))

# Distributions
plot_distribution(data['log_cpy'],'Log (Citations + 1)')
plot_distribution(data['log_alt'], 'Log (Altmetric Score + 1)')
plot_distribution(data['log_patent_mentions'], 'Log (Patent Mentions + 1)')
plot_distribution(data['log_policy_mentions'], 'Log (Policy Mentions + 1)')
plot_distribution(data['word_count'], 'Abstract Word Count')
plot_distribution(data['title_word_count'], 'Title Word Count')
plot_distribution(np.log(data['number_authors']), 'Log (Number Authors)')
plot_frequency('article_type', 'Article Type')
plot_frequency('country', 'Country')


# Correlations
plot_relationship(data, 'log_cpy', 'log_alt', 'Log(Citations + 1)', 'Log(Altmetric Score + 1)')
plot_relationship(data, 'log_cpy', 'log_patent_mentions', 'Log(Citations + 1)', 'Log(Patent Mentions + 1)')
plot_relationship(data, 'log_cpy', 'log_policy_mentions', 'Log(Citations + 1)', 'Log(Policy Mentions + 1)')
plot_relationship(data, 'log_alt', 'log_patent_mentions', 'Log(Altmetric Score + 1)', 'Log(Patent Mentions + 1)')
plot_relationship(data, 'log_alt', 'log_policy_mentions', 'Log(Altmetric Score + 1)', 'Log(Policy Mentions + 1)')
plot_relationship(data, 'log_patent_mentions', 'log_policy_mentions', 'Log(Patent Mentions + 1)', 'Log(Policy Mentions + 1)')

##############################################################################################################################
# Normalise by age 

data = normalise_by_age(data, 'log_cpy', 'Log (Citations + 1)') # Normalise citations
data = normalise_by_age(data, 'log_alt', 'Log (Altmetric Score + 1)') # Normalise altmetric score
data = normalise_by_age(data, 'log_policy_mentions', 'Log (Policy Mentions + 1)') # Normalise altmetric score
data = normalise_by_age(data, 'log_patent_mentions', 'Log (Patent Mentions + 1)') # Normalise altmetric score

# Save as CSV
data.to_csv("2_Data_Processing/preprocessed_normalised_impact.csv", index=False)
