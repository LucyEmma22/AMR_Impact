#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import chain
import requests
import json
import os
os.chdir("/Users/s1995754/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD Year 3/AMR Impact")

# Parameters
email = 'enter_email_key_here'
tool = 'lbsearch'
api_key = 'enter_api_key_here'
base_url_eSearch = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term='
parameters = '&retstart=0&retmax=9999&retmode=json&email=' + email + '&tool=' + tool + '&api_key=' + api_key #+ '&usehistory=y'

##############################################################################
# Search Term and Filters

search_term = 'drug resistance, microbial[MeSH Terms]'
date_range = '"1940/01/01"[Date - MeSH] : "2023/09/01"[Date - MeSH]'
filters =  'fha[Filter] AND english[Filter]' # fha gets only papers with an abstract

##############################################################################
# Microbial resistance with filters

# Get total number of results 
pmid_search_results = requests.get(base_url_eSearch + search_term + ' AND ' + date_range + ' AND ' + filters + ' AND ' + parameters)
pmid_search_results = json.loads(pmid_search_results.text)
total_result_count = int(pmid_search_results['esearchresult']['count']) 

# Loop though year of publication to get list of PMIDs using eSearch 
nested_pmid_list = []
year = 1940
while year <= 2023:
    date_term = '("' + str(year) + '"[Date - MeSH] : "' + str(year) + '"[Date - MeSH]) AND '
    pmid_search_results = requests.get(base_url_eSearch + date_term + search_term + ' AND ' + date_range + ' AND ' + filters + ' AND ' + parameters)
    pmid_search_results = json.loads(pmid_search_results.text)
    nested_pmid_list.append(pmid_search_results['esearchresult']['idlist'])
    year = year + 1
pmid_list = list(set(list(chain.from_iterable(nested_pmid_list))))
print(len(pmid_list))

# Write pmid list to .txt file
with open(r'1_Data_Retrieval/pmid_list2.txt', 'w') as fp:
    fp.write('\n'.join(pmid_list))
    
# Search for pmid_list2.txt was done on 12/09/23, length=130906

##############################################################################
# Microbial resistance without filters

# Get total number of results 
amr_search_results = requests.get(base_url_eSearch + search_term + ' AND ' + date_range + ' AND ' + parameters)
amr_search_results = json.loads(amr_search_results.text)
total_result_count = int(amr_search_results['esearchresult']['count']) 

# Loop though year of publication to get list of PMIDs using eSearch 
nested_pmid_list = []
year = 1940
while year <= 2023:
    for month in [1, 5, 9]:
        try:
            date_term = '("' + str(year) + '-' + str(month) + '"[Date - MeSH] : "' + str(year) + '-' + str(month+3) + '"[Date - MeSH]) AND '
            pmid_search_results = requests.get(base_url_eSearch + date_term + search_term + ' AND ' + date_range + ' AND ' + parameters)
            pmid_search_results = json.loads(pmid_search_results.text)
            nested_pmid_list.append(pmid_search_results['esearchresult']['idlist'])
        except:
            nested_pmid_list = nested_pmid_list
    year = year + 1
amr_citations = list(set(list(chain.from_iterable(nested_pmid_list))))
print(len(amr_citations))

# Write pmid list to .txt file
with open(r'1_Data_Retrieval/amr_citations_list.txt', 'w') as fp:
    fp.write('\n'.join(amr_citations))

# Search for amr_citations_list.txt was done on 12/09/23, length=183494
