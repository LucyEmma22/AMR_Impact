#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import re
import ast
import unidecode

########################################################################################################

# Function to get abstract text and labels from the xml
def get_abstract(xml): 
    content = BeautifulSoup(xml, 'xml') # Get the abstract in XML format
    abstract_texts = content.find_all('AbstractText') # Find all AbstractText elements
    abstract_list = [] # Clean the text for all AbstractText elements
    for abstract_text in abstract_texts:
        for tag in abstract_text.find_all():
            tag.unwrap()
        cleaned_text = abstract_text.get_text()
        abstract_list.append(cleaned_text)  
    labels = [tag['Label'] for tag in content.find_all('AbstractText', attrs={'Label': True})] #if content.find("AbstractText") else []
    nlm_cats = [tag['NlmCategory'] for tag in content.find_all('AbstractText', attrs={'NlmCategory': True})] #if content.find("AbstractText") else []
    lab_nlm = [(tag['Label'], tag.get('NlmCategory', '')) for tag in content.find_all('AbstractText', attrs={'Label': True})]
    language = [tag['Language'] for tag in content.find_all('OtherAbstract', attrs={'Language': True})] if content.find("OtherAbstract") else []
    typ = [tag['Type'] for tag in content.find_all('OtherAbstract', attrs={'Type': True})] if content.find("OtherAbstract") else []    
    return [abstract_list, labels, nlm_cats, lab_nlm, language, typ]


# Function to convert unicode to ASCII
def unidecode_text(text): 
    try:
        text = unidecode.unidecode(text)
    except:
        pass
    return text


# Function to converts POS tags to a form that WordNetLemmatizer() understands
def penn_to_wn(tag): 
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
    

def text_preprocess1(doc): 
    doc = re.split(r'Copyright ©|Funded by|Communicated by', doc, 1)[0] # remove all text after copyright or funding information
    doc = doc.replace("β", "beta") # replace 'β' for 'beta'
    doc = unidecode_text(doc) # convert all other unicode to ASCII
    doc = doc.lower() # transform text to lower case
    
    headings = ['.importances',
     '.areas',
     '.aims', 
     '.objectives',
     '.materials',
     '.methods',
     '.patients',
     '.results',
     '.conclusions']
    for heading in headings:
        doc = doc.replace(heading, '') # Remove headings
        
    headings = ['.importance',
     '.hypothesis',
     '.aim',
     '.objective',
     '.material',
     '.methodology',
     '.result'
     '.conclusion',
     '.key',
     '.data']
    for heading in headings:
        doc = doc.replace(heading, '') # Remove headings
        
    pattern = r'\(([^\)]+)\)|\{([^\}]+)\}|\[([^\]]+)\]' # brackets
    matches = re.findall(pattern, doc) # find all brackets
    str_in_brackets = [match[0] or match[1] or match[2] for match in matches] # get all strings in brackets
    str_in_brackets_to_remove = [s for s in str_in_brackets if any(keyword.lower() in s.lower() for keyword in ['nct0', 'isrctn', 'abstract truncated'])] # get strings in brackets containing these words
    for string in str_in_brackets_to_remove:
        doc = doc.replace(string, '') # remove strings in brackets containing these words  
        
    url_pattern = r'https?://\S+|www\.\S+'
    doc = re.sub(url_pattern, '', doc) # Remove links 
    
    sentences = re.split(r'(?<=[.!?])\s+', doc)  # Split into sentences based on punctuation
    sentences_to_remove = [s for s in sentences if any(keyword.lower() in s.lower() for keyword in ['nct0', 'isrctn'])] # get strings in brackets containing these words
    for sentence in sentences_to_remove:
        doc = doc.replace(sentence, '') # remove sentences containing these words
    pattern = re.compile(r'[\d\W_]+') # numbers and non-word characters (punctuation)
    
    doc = pattern.sub(' ', doc)  # remove numbers and non-word characters (punctuation)    
    doc = word_tokenize(doc) # tokenize (convert string to list of words) 
    doc = [x for x in doc if x not in stopwords.words('english')] # remove stop words 
    doc = [x for x in doc if len(x) > 2] # remove words of length 1 or less
    doc = nltk.pos_tag(doc) # position tagging
    defTags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS', 'JJR', 'RB', 'RBS', 'RBR', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] # list of word types to leave in the text
    doc = [(t[0], t[1]) for t in doc if t[1] in defTags] # remove words which are not a noun, adjective, verb or adverb
    doc = [(t[0], penn_to_wn(t[1])) for t in doc] # convert position tags to a form that WordNetLemmatizer() understands
    wnl = WordNetLemmatizer() # define a lemmatizing method       
    doc = [wnl.lemmatize(t[0], t[1]) for t in doc] # lemmatize
    doc = ' '.join([str(elem) for elem in doc]) # convert back to string
    return doc

# Function to preprocess mesh and article type
def text_preprocess2(doc): 
    doc = doc.replace("β", "beta") # replace beta
    doc = unidecode_text(doc) # convert unicode to ASCII
    doc = doc.lower() # transform text to lower case
    doc = ast.literal_eval(doc) # convert string to list of mesh terms
    pattern = re.compile(r'[\d\W_]+') # numbers and non-word characters (punctuation)
    doc = [pattern.sub('', element) for element in doc] # remove numbers and non-word characters (punctuation)
    doc = ' '.join([str(elem) for elem in doc]) # convert back to string
    return doc

########################################################################################################

data = pd.read_csv('extracted_data.csv') # Read in data
data.abstract.fillna(data.other_abstract, inplace=True) # If there is no abstract, replace with 'Other Abstract'
data = data.dropna(subset = ['abstract']) # Remove rows that have no abstract or 'Other Abstract'
data = data[data['mesh_terms'].apply(lambda x: len(str(x).split()) > 1)] # Remove rows where there are no mesh terms
data[['abstract_list', 'labels', 'nlm_cats', 'lab_nlm','language', 'type']] = data['abstract'].apply(lambda xml: pd.Series(get_abstract(xml))) # Use the function to get abstract text and labels from xml
data = data.drop_duplicates('abstract') # Remove duplicate abstracts

########################################################################################################
########################################################################################################
# # Trying to decide which labels should be removed from the abstract
# label_list = pd.DataFrame(data['labels'].explode('labels')).dropna(subset = ['labels']).drop_duplicates()['labels'].tolist() # List of all Labels

# #data['diff1'] =  data['labels'].apply(lambda x: len(x)) - data['abstract_list'].apply(lambda x: len(x)) # Is the number of labels the same as the number of abstract texts?
# #data['diff2'] =  data['labels'].apply(lambda x: len(x)) - data['nlm_cats'].apply(lambda x: len(x)) # Is the number of labels the same as the number of NLM categories?
# data['lab_nlm'] = data['lab_nlm'].apply(lambda x: [(item[0], '') if i != 0 and item[1] == 'BACKGROUND' else item
#                                              for i, item in enumerate(x)]) # If the NLMCatagory 'BACKGROUND' isn't first, change it to blank 
# lab_nlm_df = pd.DataFrame(data['lab_nlm'].explode('lab_nlm')).dropna(subset = ['lab_nlm']).drop_duplicates() # Make a data frame with a single column of all unique Label-NLM pairs
# lab_nlm_df[['label', 'nlm']] = lab_nlm_df['lab_nlm'].apply(pd.Series) # Split Label and NLM into seperate columns
# lab_nlm_df = lab_nlm_df[['label','nlm']] # Select Label and NLM
# lab_nlm_df = lab_nlm_df.replace(r'^\s*$', np.nan, regex=True) # Replace empty NLM with nan
# lab_nlm_df['nlm'] = lab_nlm_df.nlm.replace('UNASSIGNED',np.nan,regex = True) # Replace unassigned NLM with nan
# mapping_dict = dict(lab_nlm_df.dropna(subset=['nlm']).values) # Create a mapping dictionary from non-empty values in Label to NLM
# lab_nlm_df['nlm'] = lab_nlm_df['nlm'].fillna(lab_nlm_df['label'].map(mapping_dict)) # Fill empty rows in NLM using the mapping dictionary
# lab_nlm_df = lab_nlm_df.drop_duplicates()

# unassigned_nlm = lab_nlm_df[lab_nlm_df['nlm'].isna()] # Filter rows which have labels but no NLM
# unassigned_nlm_list = unassigned_nlm['label'].tolist() # Create a list of Labels which have no NLM associated

# data['index'] = data['labels'].apply(lambda x: [i for i, val in enumerate(x) if val in unassigned_nlm_list]) # Create a column which gives the position of non-NLM-associated labels in the list
# data['remove_label'] = data.apply(lambda row: [row['labels'][i] for i in row['index']], axis=1) # Create a column which gives the Labels at these positions 
# data['remove_text'] = data.apply(lambda row: [row['abstract_list'][i] for i in row['index']], axis=1) # Create a column which gives the text at these positions
# Label_Test = data[data['index'].apply(lambda x: bool(x))][['PMID', 'remove_label']].explode('remove_label')
# Label_Test['remove_text'] = data[data['index'].apply(lambda x: bool(x))]['remove_text'].explode('remove_text').tolist()
# Label_Test.to_csv("2_Data_Processing/Label_Test.csv", index=False) # Save as a csv to investigate these labels and decide which should be removed
# data.drop(['remove_label', 'remove_text'], axis=1, inplace=True)

########################################################################################################
########################################################################################################

# Labels I have decided to remove:
to_remove = [
 'FUNDING',
 'TRIAL REGISTRATION',
 'CONTACT',
 'SUPPLEMENTARY INFORMATION',
 'LAY SUMMARY',
 'ETHICS AND DISSEMINATION',
 'TRIAL REGISTRATION NUMBER',
 'CITATION',
 'FINANCIAL DISCLOSURE(S)',
 'KEY WORDS',
 'SPONSORS',
 'AVAILABILITY AND IMPLEMENTATION',
 'SYSTEMATIC REVIEW REGISTRATION',
 'CLINICAL TRIALS REGISTRATION',
 'TRIAL REGISTRY NAME:',
 'DATABASE',
 'PRIMARY FUNDING SOURCE',
 'FINANCIAL DISCLOSURE',
 'PROSPERO REGISTRATION NUMBER',
 'TRIAL REGISTRY',
 'GOV IDENTIFIERS',
 'ETHICS APPROVAL AND DISSEMINATION',
 'AVAILABILITY',
 'PRIMARY FUNDING SOURCES',
 'TRAIL REGISTRATION',
 'LAY ABSTRACT',
 'FROM THE CLINICAL EDITOR',
 'OSF REGISTRATION NUMBER',
 'AUTHOR CONTRIBUTION',
 'GENBANK ACCESSION NOS',
 'CLINICAL TRIAL REGISTRATION',
 'PROTOCOL REGISTRATION',
 'REGISTRATION NUMBER',
 'PROSPERO REGISTRATION',
 'STUDY REGISTRATION',
 'GOV IDENTIFIER',
 'CLINICAL TRIAL NUMBER',
 'TRIAL REGISTRATIONS',
 'SPONSOR',
 'CONFLICT OF INTEREST',
 'STUDY REGISTRATION NUMBER',
 'VIRTUAL SLIDES',
 'REGISTRATION DETAILS',
 'TRAIL REGISTRATION NUMBER',
 'ONE-SENTENCE SUMMARY',
 'SYSTEMATIC REVIEW REGISTRATION NUMBER',
 'GOVIDENTIFIERS',
 'DISCLAIMER',
 'ENDORSEMENT',
 'COTANCT',
 'REGISTRATION',
 'USEFUL WEBSITES',
 'EDITORIAL NOTE',
 'ETHICS',
 'HINTERGRUND',
 'STUDIENDESIGN',
 'ERGEBNISSE',
 'SCHLUSSFOLGERUNG',
 'STRUCTURED DIGITAL ABSTRACT'
 ]

data['index'] = data['labels'].apply(lambda x: [i for i, val in enumerate(x) if val in to_remove]) # Create a column which gives the indices of Labels to remove
data['abstract'] = data.apply(lambda row: [string for i, string in enumerate(row['abstract_list']) if i not in row['index']], axis=1) # Remove parts of abstract with Labels to remove
data.drop(['abstract_list', 'labels', 'nlm_cats', 'lab_nlm', 'language', 'type', 'index'], axis=1, inplace=True)
data['abstract'] = data['abstract'].apply(lambda x: ' '.join(x)) # Convert abstract to a string
data['sentence_count'] = data['abstract'].apply(lambda x: len(nltk.sent_tokenize(x))) # Count sentences in abstract
data['word_count'] = data['abstract'].str.split().str.len() # Count words in abstract

# Save unprocessed text as .csv 
data.to_csv("unprocessed_text.csv", index=False) 

########################################################################################################

data['p_abstract'] = data['abstract'].apply(text_preprocess1) # Process abstract text
data['p_title'] = data['title'].apply(text_preprocess1) # Process title
data['p_mesh'] = data['mesh_terms'].apply(text_preprocess2) # Process mesh
data['p_articletype'] = data['article_type'].apply(text_preprocess2) # Process article type

# Save preprocessed text as .csv 
data.to_csv("preprocessed_text.csv", index=False)
