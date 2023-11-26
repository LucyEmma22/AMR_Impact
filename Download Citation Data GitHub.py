#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
os.chdir("/Users/s1995754/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD Year 3/AMR Impact")

########################################################################################################

def get_citations(PMID):
    email = 'enter_email_here'
    tool = 'lbsearch'
    api_key = 'enter_api_key_here'
    base_url_eLink = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&linkname=pubmed_pubmed_citedin&id='
    parameters = f'&email={email}&tool={tool}&api_key={api_key}'
    url = f'{base_url_eLink}{"".join(PMID)}{parameters}'
    xml_content = requests.get(url).text
    content = BeautifulSoup(xml_content, 'xml')
    
    def get_text(element):
        return element.get_text(strip=True) if element else ""
    
    citations = []
    for link in content.find_all('Link'):
        citation = get_text(link.find("Id"))
        citations.append(citation)
    
    return [PMID, citations, len(citations)]

# Read pmid list  
with open(r'1_Data_Retrieval/pmid_list2.txt', 'r') as pmid:
    pmid_list = [line.strip() for line in pmid]

# Loop through PMID list 
citation_data = []
for PMID in pmid_list:
    print(pmid_list.index(PMID))
    try:
        citations = get_citations(PMID)
        citation_data.append(citations)    
    except:
        time.sleep(1)
        print("pause 1 sec")
        try:
            citations = get_citations(PMID)
            citation_data.append(citations)  
        except:
            time.sleep(5)
            print("pause 5 sec")
            citations = get_citations(PMID)
            citation_data.append(citations)  
            
citation_data_pd = pd.DataFrame(citation_data, columns = ['PMID', 'citations', 'number_citations'])
citation_data_pd.to_csv("1_Data_Retrieval/citation_data2.csv", index=False)

# Search for citation_data2.csv was completed on 14/09/23
