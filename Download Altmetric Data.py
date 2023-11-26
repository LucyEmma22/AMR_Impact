#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
os.chdir("/Users/s1995754/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD Year 3/AMR Impact")

# Read pmid list  
with open(r'1_Data_Retrieval/pmid_list2.txt', 'r') as pmid:
    pmid_list = [line.strip() for line in pmid]

max_length = 25000
for i in range(0, len(pmid_list), max_length):
    sublist = pmid_list[i:i + max_length]
    with open(f'1_Data_Retrieval/Altmetric_Lists/altmetric_list_{i}.txt', 'w') as fp:
        fp.write('\n'.join(sublist))

        
## SEARCH EACH LIST IN THE ALTMETRIC DATABASE AND DOWNLOAD UTF-8 CSV (Command-A (⌘A) selects entire list)
folder_path = '1_Data_Retrieval/Altmetric_Lists'
file_list = os.listdir(folder_path)
altmetric_data = pd.DataFrame()
for file in file_list:
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        print(len(df))
        altmetric_data = pd.concat([altmetric_data, df], ignore_index=True)

altmetric_data.to_csv('1_Data_Retrieval/altmetric_data2.csv', index=False)

# Search for altmetric_data2.csv was completed on 14/09/23
