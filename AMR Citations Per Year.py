#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import os
os.chdir("/Users/s1995754/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD Year 3/AMR Impact") 
plt.style.use('seaborn-v0_8-whitegrid')

data = pd.read_csv("3_Data_Analysis/Number_AMR_by_Year.csv")
data = data[data['Year']<=2020]
data = data[data['Year']>=1900]

fig, ax = plt.subplots(figsize=(6, 4))
fontsize = 10
plt.plot(data['Year'], data['Count'], linewidth=2, c="black")
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.title('Number Of PubMed Publications Relating To AMR Over Time')
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
fig.tight_layout()
plt.savefig('Figures/Number Of PubMed Publications Relating To AMR Over Time.pdf', format='pdf')
plt.show()
