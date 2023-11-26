#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:35:43 2023

@author: s1995754
"""
# Import libraries
import pandas as pd
import re
import numpy as np
import os
os.chdir("/Users/s1995754/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD Year 3/AMR Impact")

########################################################################################################

def find_here_sentences(abstract):
    here = re.findall(r'\b(Here[\s,]+we [^.]*\.)', abstract)
    here = ' '.join([str(elem) for elem in here]) 
    return here

data = pd.read_csv("2_Data_Processing/preprocessed_normalised_impact.csv") # Import data
data["here_sentence"] = data["abstract"].apply(find_here_sentences) # Find here sentences
data['here_presence'] = data['here_sentence'].apply(lambda x: 0 if x == "" else 1) # Binary column to say if there is a here sentence
data['contains_nature'] = data['journal'].str.contains('nature', case=False).astype(int) # Binary column to say if it is a nature paper
sum(data['contains_nature']) # Number of nature papers in data set
sum(data[data['contains_nature']==1]['here_presence']) / len(data[data['contains_nature']==1]['here_presence']) # Proportion of nature papers with a 'here' sentence
sum(data[data['contains_nature']==0]['here_presence']) / len(data[data['contains_nature']==0]['here_presence']) # Proportion of non-nature papers with a 'here' sentence
nature_here_sentences = data[(data['contains_nature'] == 1) & (data['here_presence'] == 1)]['here_sentence'] # All here sentences in nature papers

np.mean(data[data['contains_nature']==1]['log_cpy_residual']) # Mean citations per year for nature papers
np.mean(data[data['contains_nature']==1]['log_alt_residual']) # Mean altmetric score for nature papers
np.mean(data[data['contains_nature']==1]['log_policy_mentions_residual']) # Mean policy inclusions for nature papers
np.mean(data[data['contains_nature']==1]['log_patent_mentions_residual']) # Mean patent inclusions for nature papers

np.mean(data[data['contains_nature']==0]['log_cpy_residual']) # Mean citations per year for non-nature papers
np.mean(data[data['contains_nature']==0]['log_alt_residual']) # Mean altmetric score for non-nature papers
np.mean(data[data['contains_nature']==0]['log_policy_mentions_residual']) # Mean policy inclusions for non-nature papers
np.mean(data[data['contains_nature']==0]['log_patent_mentions_residual']) # Mean patent inclusions for non-nature papers
