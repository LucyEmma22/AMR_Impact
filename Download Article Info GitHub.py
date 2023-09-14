#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

########################################################################################################

def get_article_info(PMID):
    email = enter_email_address_here
    tool = enter_tool_name_here
    api_key = enter_api_key_here
    base_url_eFetch = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id='
    parameters = f'&email={email}&tool={tool}&api_key={api_key}'
    url = f'{base_url_eFetch}{"".join(PMID)}{parameters}'
    xml_content = requests.get(url).text
    content = BeautifulSoup(xml_content, 'xml')
    
    def get_text(element):
        return element.get_text(strip=True) if element else ""
    
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


# Read pmid list  
with open(r'pmid_list.txt', 'r') as pmid:
    pmid_list = [line.strip() for line in pmid]

# Loop through PMID list 
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

# Save extracted data as .csv file
extracted_data_pd = pd.DataFrame(extracted_data, columns = ['PMID', 'title', 'abstract', 'other_abstract', 'journal', 'country', 'year', 'month', 'day', 'mesh_terms', 'mesh_terms_q', 'article_type'])
extracted_data_pd.to_csv("extracted_data.csv", index=False)