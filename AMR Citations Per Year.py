#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import os
os.chdir("/Users/s1995754/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD Year 3/AMR Impact") # set working directory
plt.style.use('seaborn-v0_8-whitegrid')
set_matplotlib_formats('svg')


data = pd.read_csv("3_Data_Analysis/Prop_AMR_by_Year.csv")
data = data[data['year']<=2019]
data = data[data['year']>=1900]

plt.plot(data['year'], data['amr']/data['journal_article'], linewidth=2, c="black")
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.title('Proportion Of PubMed Journal Articles Relating To AMR')
plt.show()

data = pd.read_csv("3_Data_Analysis/Number_AMR_by_Year.csv")
data = data[data['Year']<=2020]
data = data[data['Year']>=1900]

plt.plot(data['Year'], data['Count'], linewidth=2, c="black")
plt.xlabel('Year')
plt.ylabel('Number of Citations')
plt.title('Number Of PubMed Citations Relating To AMR Over Time')
plt.show()
