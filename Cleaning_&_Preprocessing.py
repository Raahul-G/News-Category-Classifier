# Importing Libraries

import pandas as pd
import numpy as np

# Loading the data

path = "News_Category_Dataset_v2.json"

data = pd.read_json(path, lines=True)

# Checking NUll values
data.isnull().any()

# Dropping Unwanted-Columns
data.drop(['authors', 'link', 'date'], inplace=True, axis=1)

# Indexing Unwanted-categories
women = data[data['category'] == 'WOMEN'].index.values
qv = data[data['category'] == 'QUEER VOICES'].index.values
fifty = data[data['category'] == 'FIFTY'].index.values
hl = data[data['category'] == 'HOME & LIVING'].index.values
wn = data[data['category'] == 'WEIRD NEWS'].index.values
lv = data[data['category'] == 'LATINO VOICES'].index.values
impact = data[data['category'] == 'IMPACT'].index.values

# Removing Unwanted - Categories
data.drop(index=qv, inplace=True, axis=0)
data.drop(index=fifty, inplace=True, axis=0)
data.drop(index=hl, inplace=True, axis=0)
data.drop(index=wn, inplace=True, axis=0)
data.drop(index=lv, inplace=True, axis=0)
data.drop(index=impact, inplace=True, axis=0)
data.drop(index=women, inplace=True, axis=0)

# Grouping Categories
data.category = data.category.map(lambda x: "ARTS , CULTURE & ENVIRONMENT" if x == "CULTURE & ARTS" or x == 'ARTS' or x == "GREEN" or x == "ENVIRONMENT" or x == 'ARTS & CULTURE' else x)
data.category = data.category.map(lambda x: "BLACK LIVES MATTER" if x == "BLACK VOICES" else x)
data.category = data.category.map(lambda x: "BUSINESS & MONEY" if x == "BUSINESS" or x == 'MONEY' else x)
data.category = data.category.map(lambda x: "LAW & CRIME" if x == "DIVORCE" or x == 'CRIME' else x)
data.category = data.category.map(lambda x: "SPORTS & EDUCATION" if x == "EDUCATION" or x == 'TECH' or x == 'SCIENCE' or x == 'SPORTS' or x == 'COLLEGE' else x)
data.category = data.category.map(lambda x: "ENTERTAINMENT" if x == "MEDIA" or x == 'COMEDY' else x)
data.category = data.category.map(lambda x: "FOOD & TRAVEL" if x == "FOOD & DRINK" or x == 'TRAVEL' or x == 'TASTE' else x)
data.category = data.category.map(lambda x: "PARENTING" if x == "PARENTS" else x)
data.category = data.category.map(lambda x: "STYLE & BEAUTY" if x == "STYLE" else x)
data.category = data.category.map(lambda x: "WORLD NEWS" if x == "THE WORLDPOST" or x == 'WORLDPOST' else x)
data.category = data.category.map(
    lambda x: "GOOD NEWS" if x == "WEDDINGS" or x == 'HEALTHY LIVING' or x == 'WELLNESS' else x)
data.category = data.category.map(lambda x: "POLITICS & RELIGION" if x == "RELIGION" or x == 'POLITICS' else x)

# Removing Duplicated Rows
duplicate = data[data.duplicated(['short_description', 'headline', 'category'])].index.values
data.drop(index=duplicate, inplace=True, axis=0)

# Converting the texts into lower-case¶
data.headline = data['headline'].str.lower()
data.short_description = data['short_description'].str.lower()

# Finding & Removing empty columns
data.loc[data['headline'] == "", 'headline'] = np.nan
data.dropna(subset=['headline'], inplace=True)

data.loc[data['short_description'] == "", 'short_description'] = np.nan
data.dropna(subset=['short_description'], inplace=True)

# Joining headlines and Short Description
data['text'] = data['headline'] + " " + data['short_description']
data.drop(columns=['headline', 'short_description'], inplace=True, axis=1)

# Saving the cleaned data into CSV
data.to_csv('cleaned_data.csv', index=False)