#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

###############################################################################################################

# Read pmid list  
with open(r'pmid_list.txt', 'r') as pmid:
    pmid_list = [line.strip() for line in pmid]

# Split PMID list into sublists <25000 in length and save in a folder called Altmetric Lists
max_length = 25000
for i in range(0, len(pmid_list), max_length):
    sublist = pmid_list[i:i + max_length]
    with open(f'Altmetric_Lists/altmetric_list_{i}.txt', 'w') as fp:
        fp.write('\n'.join(sublist))
        
## SEARCH EACH LIST IN THE ALTMETRIC DATABASE AND DOWNLOAD UTF-8 CSV (Command-A (⌘A) selects entire list)

# Merge search results into one dataframe
folder_path = 'Altmetric_Lists'
file_list = os.listdir(folder_path)
altmetric_data = pd.DataFrame()
for file in file_list:
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        print(len(df))
        altmetric_data = pd.concat([altmetric_data, df], ignore_index=True)

# Save altmetric data as .csv
altmetric_data.to_csv('altmetric_data.csv', index=False)
