#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
os.chdir("/Users/s1995754/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD Year 3/AMR Impact")

########################################################################################################

def get_text(element):
    return element.get_text(strip=True) if element else ""

def get_article_info(PMID):
    email = 'enter_email_here'
    tool = 'lbsearch'
    api_key = 'enter_api_key_here'
    base_url_eFetch = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id='
    parameters = f'&email={email}&tool={tool}&api_key={api_key}'
    url = f'{base_url_eFetch}{"".join(PMID)}{parameters}'
    xml_content = requests.get(url).text
    content = BeautifulSoup(xml_content, 'xml')
    title = get_text(content.find("ArticleTitle"))
    abstract = content.find("Abstract")
    other_abstract = content.find("OtherAbstract")
    journal = get_text(content.find("Title"))
    country = get_text(content.find("Country"))
    article_type = [get_text(at) for at in content.find_all('PublicationType')]
    date = content.find('PubMedPubDate', {'PubStatus': 'pubmed'})
    year = get_text(date.find('Year'))
    month = get_text(date.find('Month'))
    day = get_text(date.find('Day'))    
    mesh_terms = []
    mesh_terms_q = []
    for mesh in content.find_all('MeshHeading'):
        descriptor = get_text(mesh.find("DescriptorName"))
        qualifier = get_text(mesh.find("QualifierName"))
        mesh_terms.append(descriptor)
        if descriptor and qualifier:
            mesh_terms_q.append(f"{descriptor}_{qualifier}")
        else:
            mesh_terms_q.append(descriptor)      
    return [PMID, title, abstract, other_abstract, journal, country, year, month, day, mesh_terms, mesh_terms_q, article_type]


def get_authors(PMID):
    email = 'enter_email_here'
    tool = 'lbsearch'
    api_key = 'enter_api_key_here'
    base_url_eFetch = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id='
    parameters = f'&email={email}&tool={tool}&api_key={api_key}'
    url = f'{base_url_eFetch}{"".join(PMID)}{parameters}'
    xml_content = requests.get(url).text
    content = BeautifulSoup(xml_content, 'xml')    
    authors = []
    affiliations = []
    for author in content.find_all('Author'):
        lastname = get_text(author.find("LastName"))
        firstname = get_text(author.find("ForeName"))
        affiliation = get_text(author.find("Affiliation"))
        authors.append(lastname + ', ' + firstname)
        affiliations.append(affiliation)
    affiliations = [item for item in affiliations if item != '']
    affiliations = list(dict.fromkeys(affiliations))
    number_authors = len(authors)
    number_affiliations = len(affiliations)
    if content.find('KeywordList'):
        keywords = [get_text(word) for word in content.find('KeywordList')]
    else:
        keywords = []
    return [PMID, authors, number_authors, affiliations, number_affiliations, keywords]

########################################################################################################

# Read pmid list  
with open(r'1_Data_Retrieval/pmid_list2.txt', 'r') as pmid:
    pmid_list = [line.strip() for line in pmid]

# Article Info
extracted_data = []
for PMID in pmid_list:
    print(pmid_list.index(PMID))
    try:
        info = get_article_info(PMID)
        extracted_data.append(info)    
    except:
        time.sleep(1)
        print("pause 1 sec")
        try:
            info = get_article_info(PMID)
            extracted_data.append(info)  
        except:
            time.sleep(5)
            print("pause 5 sec")
            info = get_article_info(PMID)
            extracted_data.append(info)  
            
extracted_data_pd = pd.DataFrame(extracted_data, columns = ['PMID', 'title', 'abstract', 'other_abstract', 'journal', 'country', 'year', 'month', 'day', 'mesh_terms', 'mesh_terms_q', 'article_type'])

# Authors, Affiliations and Keywords
author_list = []
for PMID in pmid_list:
    print(pmid_list.index(PMID))
    try:
        authors = get_authors(PMID)
        author_list.append(authors)    
    except:
        time.sleep(1)
        print("pause 1 sec")
        try:
            authors = get_authors(PMID)
            author_list.append(authors)  
        except:
            time.sleep(5)
            print("pause 5 sec")
            authors = get_authors(PMID)
            author_list.append(authors)  
            
authors_pd = pd.DataFrame(author_list, columns = ['PMID', 'authors','number_authors', 'affiliations','number_affiliations', 'keywords'])

all_extracted_data_pd = pd.merge(extracted_data_pd, authors_pd, on='PMID', how='outer')
all_extracted_data_pd.to_csv("1_Data_Retrieval/all_extracted_data.csv", index=False)

# Search for all_extracted_data.csv was completed on 15/09/23
