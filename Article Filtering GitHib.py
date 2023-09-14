#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
import pandas as pd

########################################################################################################

# Import data
data = pd.read_csv('preprocessed_text.csv') # Read in data

# Get a list of article type labels
# article_types = pd.DataFrame(data['article_type'].value_counts()).reset_index()
# article_types['index'] = article_types.apply(lambda row:  row['index'].strip('[]').replace('"', "'").split("',")  , axis=1)
# article_types = article_types.explode('index')
# article_types['index'] = article_types['index'].str.strip().str.strip("'")
# article_types = article_types.groupby('index', as_index=False)['article_type'].sum()
# article_types_list = article_types['index'].tolist()

# Print article_types_list and comment out what we want to keep 
article_types_to_remove = [
 #'Adaptive Clinical Trial',
 'Autobiography',
 'Bibliography', 
 'Biography',
 'Case Reports',
 'Clinical Conference',
 #'Clinical Study',
 #'Clinical Trial',
 #'Clinical Trial Protocol',
 #'Clinical Trial, Phase I',
 #'Clinical Trial, Phase II',
 #'Clinical Trial, Phase III',
 #'Clinical Trial, Phase IV',
 #'Clinical Trial, Veterinary',
 'Comment',
 #'Comparative Study',
 'Congress',
 'Consensus Development Conference',
 'Consensus Development Conference, NIH',
 #'Controlled Clinical Trial',
 #'Corrected and Republished Article',
 #'Dataset',
 'Directory',
 #'Duplicate Publication',
 'Editorial',
 #'Equivalence Trial',
 #'Evaluation Study',
 'Festschrift',
 'Guideline',
 #'Historical Article',
 'Interview',
 'Introductory Journal Article',
 #'Journal Article',
 'Lecture',
 'Legal Case',
 'Letter',
 'Meta-Analysis',
 #'Multicenter Study',
 'News',
 'Newspaper Article',
 #'Observational Study',
 #'Observational Study, Veterinary',
 'Overall',
 'Personal Narrative',
 'Portrait',
 'Practice Guideline',
 #'Pragmatic Clinical Trial',
 'Published Erratum',
 #'Randomized Controlled Trial',
 #'Randomized Controlled Trial, Veterinary',
 #'Research Support, American Recovery and Reinvestment Act',
 #'Research Support, N.I.H., Extramural',
 #'Research Support, N.I.H., Intramural',
 #"Research Support, Non-U.S. Gov't",
 #"Research Support, U.S. Gov't, Non-P.H.S.",
 #"Research Support, U.S. Gov't, P.H.S.",
 'Retracted Publication',
 'Review',
 'Systematic Review',
 'Technical Report'
 #'Twin Study',
 #'Validation Study',
 #'Video-Audio Media'
 ]

data_filtered = data[~data['article_type'].str.contains('|'.join(article_types_to_remove))] # Filter rows that contain article types we want to remove
non_english = [27992149, 29676866, 32220171, 31529848, 28671150] # These papers were identified while looking at low frequency ngrams
data_filtered = data_filtered[~data_filtered['PMID'].isin(non_english)]

# Remove rows with less than 2 words in the processed abstract, title or mesh (required for 2grams and mesh pairs)
data_filtered['p_abstract_word_count'] = data_filtered['p_abstract'].str.split().str.len() # Count words in processed abstract
data_filtered = data_filtered[data_filtered['p_abstract_word_count']>1]
data_filtered['p_title_word_count'] = data_filtered['p_title'].str.split().str.len() # Count words in processed title
data_filtered = data_filtered[data_filtered['p_title_word_count']>1]
data_filtered['p_mesh_word_count'] = data_filtered['p_mesh'].str.split().str.len() # Count words in processed mesh
data_filtered = data_filtered[data_filtered['p_mesh_word_count']>1]

# Save filtered preprocessed text as .csv 
data_filtered.to_csv("preprocessed_text_filtered.csv", index=False)