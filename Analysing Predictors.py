#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from wordcloud import WordCloud
import ast
from scipy import stats
from scipy.stats import linregress
import os
os.chdir("/Users/s1995754/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD Year 3/AMR Impact")

########################################################################################################
bins = 50

# Function to generate mesh term pairs from mesh terms
def generate_pairs(mesh): 
    pairs = []
    for i in range(len(mesh)):
        for j in range(i+1, len(mesh)):
            pairs.append([mesh[i], mesh[j]])
    return pairs


# Function to plot a word cloud 
def plot_word_cloud(data, predictor, response):
    wc_data = {word: abs(response) for word, response in zip(data[predictor], data[response])}
    wordcloud_instance = WordCloud(width = 400, height = 400, 
                background_color ='white', 
                stopwords=None,
                min_font_size = 10).generate_from_frequencies(wc_data)  
    
    fig, ax = plt.subplots(figsize=(5, 5))
      
    plt.imshow(wordcloud_instance) 
    
    plt.axis("off") 
    plt.tight_layout() 
    plt.savefig(f'Figures/{predictor}_response_wordcloud.pdf', format='pdf')
    plt.show()   

# Function to generate a dataframe with all unique values of a predictor, their mean response (citation rate or altmetric score) and their frequency
# Plots a word cloud for effect
def get_effects(data, predictor, response):
    overall_mean = np.mean(data[response])
    df = data[[predictor,response]].dropna().explode(predictor).reset_index(drop=True) # Exploding the list of words into separate rows
    value_counts = df[predictor].value_counts()
    df['raw_freq'] = df[predictor].map(value_counts)
    df = df[df['raw_freq']>=10]
    df = df.groupby(predictor, as_index=False).agg({response: ['mean', lambda x: stats.sem(x)], 'raw_freq': 'first'})
    df.columns = [predictor, 'mean_response', 'SEM_response', 'raw_freq'] 
    df['t_stat'] = (df['mean_response'] - overall_mean) / df['SEM_response']
    df.sort_values(by='t_stat', ascending=False, inplace=True)
    #plot_word_cloud(df[df['t_stat']>0], predictor, 't_stat')
    return df


# Function to calculate the mean response score (citation rate or altmetric score) for a predictor
def calculate_scores(data, predictor):
    df = data[predictor].explode().value_counts().reset_index() # Exploding the list of words into separate rows
    df.columns = [predictor, 'raw_freq'] 
    df['freq'] = df['raw_freq']/data[predictor].count() 
    df_dict = dict(zip(df[predictor], df['freq']))
    data2 = data[['PMID', predictor]]
    data2[f'{predictor}_uniqueness_score'] = [[df_dict[i] for i in j] for j in data2[predictor]]
    data2[f'{predictor}_uniqueness_score'] = [sum(i)/len(i) for i in data2[f'{predictor}_uniqueness_score']]
    return data2[['PMID', f'{predictor}_uniqueness_score']]


# Generates a data frame of the change in response with uniqueness and change in variance of response with uniqueness
# Plots a density plot showing relationship between uniqueness and response
def uniqueness_relationship(data, predictor, response, xlab, ylab):
    data2 = data[[response, f'{predictor}_uniqueness_score']].dropna()
    model = np.polyfit(data2[f'{predictor}_uniqueness_score'], data2[response], 1)
    model_predict = np.polyval(model, data2[f'{predictor}_uniqueness_score'])
    line_xvals = np.linspace(data2[f'{predictor}_uniqueness_score'].min(), data2[f'{predictor}_uniqueness_score'].max(), 100)
    line_yvals = np.polyval(model, line_xvals)
    slope, intercept, r_value, p_value, std_err = linregress(data2[f'{predictor}_uniqueness_score'], data2[response])
    
    # Density Plot
    fontsize = 16
    fig, ax = plt.subplots(figsize=(6, 5))   
    plt.hist2d(data2[f'{predictor}_uniqueness_score'], data2[response], bins=bins, cmap='viridis',norm=colors.LogNorm())
    plt.plot(line_xvals, line_yvals, c='red', linewidth = '3')
    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    specific_ticks = [0.05, 0.1, 0.15, 0.2]
    plt.xticks(specific_ticks, specific_ticks)
    plt.colorbar(label='Density')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(f'Figures/Similarity Score_{predictor}_{response}.pdf', format='pdf')
    plt.show()

    data2['model_predict'] = model_predict
    least_similar = data2[data2[f'{predictor}_uniqueness_score']==min(data2[f'{predictor}_uniqueness_score'])].reset_index(drop=True)['model_predict'][0]
    most_similar = data2[data2[f'{predictor}_uniqueness_score']==max(data2[f'{predictor}_uniqueness_score'])].reset_index(drop=True)['model_predict'][0]
    percentile_least_similar = (len(data2[data2[response]<least_similar])/len(data2))*100  
    percentile_most_similar = (len(data2[data2[response]<most_similar])/len(data2))*100
    
    # Linear regression of square root absolute residuals
    residuals = np.sqrt(np.abs(model_predict - data2[response])) # Square root of absolute residuals
    model2 = np.polyfit(data2[f'{predictor}_uniqueness_score'], residuals, 1) # Linear regression model for residuals
    model2_predict = np.polyval(model2, data2[f'{predictor}_uniqueness_score'])
   
    line_xvals = np.linspace(data2[f'{predictor}_uniqueness_score'].min(), data2[f'{predictor}_uniqueness_score'].max(), 100)
    line_yvals = np.polyval(model2, line_xvals)
    
    slope_residuals, intercept_residuals, r_value_residuals, p_value_residuals, std_err_residuals = linregress(data2[f'{predictor}_uniqueness_score'], residuals)

    # Percentage change in residuals from least to most unique
    data2['model2_predict'] = model2_predict
    least_similar = data2[data2[f'{predictor}_uniqueness_score']==min(data2[f'{predictor}_uniqueness_score'])].reset_index(drop=True)['model2_predict'][0]
    most_similar = data2[data2[f'{predictor}_uniqueness_score']==max(data2[f'{predictor}_uniqueness_score'])].reset_index(drop=True)['model2_predict'][0]
    percentage_change = ((least_similar - most_similar) / most_similar) * 100
    
    # Result
    result = pd.DataFrame({
        'predictor': [predictor],
        'response': [response],
        'percentile_least_similar': [percentile_least_similar],
        'percentile_most_similar': [percentile_most_similar],
        'change': [percentile_least_similar - percentile_most_similar],
        'percentage_change': [percentage_change],
        'p_value_main_model': [p_value],
        'p_value_residuals_model': [p_value_residuals],
    })

    return result   
  
########################################################################################################

# Import data 
all_data = pd.read_csv("2_Data_Processing/preprocessed_normalised_impact.csv")

# Make abstract and title ngrams
all_data['abstract_1grams'] = all_data['p_abstract'].apply(lambda x: list(set(nltk.word_tokenize(x))))
all_data['title_1grams'] = all_data['p_title'].apply(lambda x: list(set(nltk.word_tokenize(x))))

# Convert mesh terms to a list and make Mesh Pairs
all_data['mesh_terms'] = all_data['mesh_terms'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else []).apply(lambda x: sorted(x)) # Convert string of mesh terms to list of mesh terms and sort alphabetically
all_data['mesh_pairs'] = all_data['mesh_terms'].apply(generate_pairs) # Make mesh term pairs
all_data['mesh_pairs_str'] = all_data['mesh_pairs'].apply(lambda x: ['_'.join(lst) for lst in x]) # Make mesh pair string

# Convert article types to a list
all_data['article_type'] = all_data['article_type'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else []) # Convert string of article types to list of mesh terms

data_new = all_data[all_data['year']>=2020]
data = all_data[all_data['year']<2020]

########################################################################################################

response_list = ['log_cpy_residual', 'log_alt_residual', 'log_patent_mentions_residual', 'log_policy_mentions_residual']
predictor_list1 = ['abstract_1grams', 'title_1grams', 'mesh_terms', 'article_type']
ylabs = ['Mean Citations', 'Mean Altmetric Score', 'Mean Patent Mentions', 'Mean Policy Mentions']
xlabs = ['Abstract', 'Title', 'MeSH Terms', 'Article Type']

# Get predictors with greatest effect
high_impact = pd.DataFrame()
for predictor, xlab in zip(predictor_list1, xlabs):
    for response, ylab in zip(response_list, ylabs):
        df = get_effects(data, predictor, response)
        high_impact[f'{predictor}_{response}'] = df[0:10][predictor].tolist()
        exec(f'{predictor}_{response} = df')

        fig, ax = plt.subplots(figsize=(8, 6))
        fontsize = 14
        a = df[0:10][predictor]
        max_words_per_line = 1
        wrapped_labels = ['\n'.join(' '.join(label.split()[i:i+max_words_per_line]) for i in range(0, len(label.split()), max_words_per_line)) for label in a]
        plt.bar(wrapped_labels, df[0:10]['mean_response'], color ='blue', width = 0.6, yerr=df[0:10]['SEM_response'], capsize=5, alpha=0.5)
        plt.axhline(y=np.mean(df['mean_response']), color='black', linestyle='--')
        
        plt.xlabel(xlab, fontsize=fontsize)
        plt.ylabel(ylab,fontsize=fontsize)
        plt.xticks(rotation = 90)
        
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.tight_layout()
        plt.savefig(f'Figures/Test/{predictor}_{response}_barchart.pdf', format='pdf')
        plt.show()
        
high_impact.to_csv('high_impact_words.csv')

# Get predictors with greatest effect - post 2020 
high_impact_new = pd.DataFrame()
for predictor, xlab in zip(predictor_list1, xlabs):
    for response, ylab in zip(response_list, ylabs):
        df = get_effects(data_new, predictor, response)
        high_impact_new[f'{predictor}_{response}'] = df[0:10][predictor].tolist()
        exec(f'{predictor}_{response}_new = df')

        fig, ax = plt.subplots(figsize=(8, 6))
        fontsize = 14
        a = df[0:10][predictor]
        max_words_per_line = 1
        wrapped_labels = ['\n'.join(' '.join(label.split()[i:i+max_words_per_line]) for i in range(0, len(label.split()), max_words_per_line)) for label in a]
        plt.bar(wrapped_labels, df[0:10]['mean_response'], color ='blue', width = 0.6, yerr=df[0:10]['SEM_response'], capsize=5, alpha=0.5)

        plt.xlabel(xlab, fontsize=fontsize)
        plt.ylabel(ylab,fontsize=fontsize)
        plt.xticks(rotation = 90)
        
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        fig.tight_layout()
        plt.savefig(f'Figures/New/{predictor}_{response}_new_barchart.pdf', format='pdf')
        plt.show()
        
high_impact_new.to_csv('high_impact_words_new.csv')


# Calculate uniqueness scores and plot relationship with response (all data)
predictor_list2 = ['abstract_1grams', 'title_1grams', 'mesh_pairs_str']
xlabs = ['Abstract Similarity Score', 'Title Similarity Score', 'MeSH Pair Similarity Score']
ylabs = ['Citations', 'Altmetric Score', 'Patent Mentions', 'Policy Mentions']

uniqueness_results = pd.DataFrame()
for predictor, xlab in zip(predictor_list2, xlabs):
    all_data = all_data.merge(calculate_scores(all_data, predictor), how='left')
    for response, ylab in zip(response_list, ylabs):
        uniqueness_results = pd.concat([uniqueness_results, uniqueness_relationship(all_data, predictor, response, xlab, ylab)])

uniqueness_results.to_csv('3_Data_Analysis/uniqueness_results.csv', index=False) 
uniqueness_scores = all_data.loc[:, all_data.columns.str.contains('uniqueness|PMID')]
uniqueness_scores.to_csv('3_Data_Analysis/uniqueness_scores.csv', index=False) 
