#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
import matplotlib.colors as colors
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import os
os.chdir("/Users/s1995754/Library/CloudStorage/OneDrive-UniversityofEdinburgh/PhD Year 3/AMR Impact")

#####################################################################################
# Functions
bins = 50

class Selector1(BaseEstimator, TransformerMixin): 
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None, *parg, **kwarg):
        return self
    def transform(self, X):
        return X[self.key] # returns a Pandas Series
    
class Selector2(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.key]] # returns a Pandas DataFrame

def feature_preprocess(text_features, numeric_features, categorical_features):
    vec_tdidf = TfidfVectorizer(ngram_range=(1,1), analyzer='word', norm='l2')
    text_transformers = FeatureUnion(transformer_list=[('{}_text'.format(feature), Pipeline([('selector', Selector1(key=feature)),('vectorizer', vec_tdidf)])) for feature in text_features])
    numeric_transformers = FeatureUnion(transformer_list=[('{}_numeric'.format(feature), Pipeline([('selector', Selector2(key=feature))])) for feature in numeric_features])
    categorical_transformers = FeatureUnion(transformer_list=[('{}_categorical'.format(feature), Pipeline([('selector', Selector2(key=feature)),('encoder', OneHotEncoder(handle_unknown='ignore'))])) for feature in categorical_features])
    preprocessor = ColumnTransformer(transformers=[
        ('text', text_transformers, text_features),
        ('numeric', numeric_transformers, numeric_features),
        ('categorical', categorical_transformers, categorical_features)])
    preprocessor.fit(all_data[text_features + numeric_features + categorical_features])
    X_all = preprocessor.transform(all_data[text_features + numeric_features + categorical_features])
    return X_all

def split_data(X_all, response):
    y_all = all_data[response]
    data_old = all_data[all_data['year'] < 2020].dropna(subset=response)
    data_new = all_data[all_data['year'] >= 2020].dropna(subset=response)
    X = X_all[data_old.index]
    y = y_all[data_old.index]
    X_new = X_all[data_new.index] 
    y_new = y_all[data_new.index] 
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_validation, y_train, y_validation = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)
    return X_new, y_new, X_train, X_validation, y_train, y_validation, X_test, y_test

def regression_model():
    xgb_params = { 
        'n_estimators': 1000, # 100
        'subsample': 1, # 1
        'min_child_weight': 1, # 1
        'max_depth': 6, # 6
        'learning_rate': 0.3, # 0.3
        'lambda': 1, # 1
        'gamma': 0, # 0
        'colsample_bytree': 1, # 1
        'alpha': 0, # 0
        'early_stopping_rounds': 10, 
        'eval_metric': 'rmse'}
    reg = XGBRegressor(**xgb_params, objective='reg:squarederror', seed=42)
    reg.fit(X_train, y_train, eval_set=[(X_validation, y_validation)])
    r2 = metrics.r2_score(y_test, reg.predict(X_test))
    return reg, r2

def regression_tuning(param_grid):
    xgb_params = { 
        'n_estimators': 1000, # 100
        'subsample': 1, # 1
        'min_child_weight': 1, # 1
        'max_depth': 6, # 6
        'learning_rate': 0.3, # 0.3
        'lambda': 1, # 1
        'gamma': 0, # 0
        'colsample_bytree': 1, # 1
        'alpha': 0, # 0
        'early_stopping_rounds': 10, 
        'eval_metric': 'rmse'}
    reg = XGBRegressor(**xgb_params, objective='reg:squarederror', seed=42)
    grid_search = GridSearchCV(estimator = reg, param_grid = param_grid, scoring='r2',cv = 3, n_jobs = 15, verbose = 0, return_train_score=True)
    grid_search.fit(X_train, y_train, eval_set=[(X_validation, y_validation)])
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    r2 = metrics.r2_score(y_test, best_estimator.predict(X_test))
    return best_estimator, best_params, r2

def print_stats_reg(preds, target, name, impact, title):
    R2 = round(metrics.r2_score(target, preds), 3)
    RMSE = round(np.sqrt(metrics.mean_squared_error(target, preds)), 3)
    fontsize = 16
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.hist2d(target, preds, bins=bins, cmap='viridis',norm=colors.LogNorm())
    plt.xlabel(f'True {impact}',fontsize=fontsize)
    plt.ylabel(f'Predicted {impact}',fontsize=fontsize)
    plt.title(f'{title} \nRMSE = {RMSE} ; R2 = {R2}', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.colorbar(label='Density')
    fig.tight_layout()
    plt.savefig(f'Figures/{name}.pdf', format='pdf')
    plt.show()
    

def regression_model_output(model,name, impact, title):
    print_stats_reg(model.predict(X_test), y_test, f'Test/model_{name}_test', impact, f'{title} (Test)')
    print_stats_reg(model.predict(X_new), y_new, f'New/model_{name}_new', impact, f'{title} (New)')

#####################################################################################
# Read in Data
all_data = pd.read_csv('2_Data_Processing/preprocessed_normalised_impact.csv')
uniqueness_scores = pd.read_csv('3_Data_Analysis/uniqueness_scores.csv')
all_data = pd.merge(all_data, uniqueness_scores, on='PMID', how='left')
Xa = feature_preprocess(['p_articletype'] , ['word_count', 'title_word_count', 'number_authors'], ['country'])
Xb = feature_preprocess(['p_abstract', 'p_title', 'p_mesh', 'p_articletype'] , ['word_count', 'title_word_count', 'number_authors'], ['country'])
Xc = feature_preprocess(['p_articletype'] , ['word_count', 'title_word_count', 'number_authors', 'abstract_1grams_uniqueness_score', 'title_1grams_uniqueness_score', 'mesh_pairs_str_uniqueness_score'], ['country'])

#####################################################################################
#####################################################################################
# HYPERPARAMETER TUNING ROUNDS
# 1 - 'learning_rate':[0.01, 0.05, 0.1, 0.3] 
# 2 - 'max_depth': [3, 6, 10], 'min_child_weight': [1, 5, 10] 
# 3 - 'subsample': [0.6, 0.8, 1],'colsample_bytree': [0.6, 0.8, 1] 
# 4 - 'gamma':[0, 0.1, 0.2, 0.3] 
# 5 - 'lambda': [0, 1, 2], 'alpha': [0, 1, 2] 
#####################################################################################
#####################################################################################
# BASE RMSE

responselist = ['log_cpy_residual', 'log_alt_residual', 'log_patent_mentions_residual', 'log_policy_mentions_residual']
RMSE_base_test = []
RMSE_base_new = []
for i in responselist:
    predictor = Xa
    X_new, y_new, X_train, X_validation, y_train, y_validation, X_test, y_test = split_data(predictor, i)
    mean_target_test = np.full_like(y_test, np.mean(y_test))
    RMSE_base_test.append(np.sqrt(metrics.mean_squared_error(y_test, mean_target_test)))
    mean_target_new = np.full_like(y_new, np.mean(y_new))
    RMSE_base_new.append(np.sqrt(metrics.mean_squared_error(y_new, mean_target_new)))
RMSE_base_df = pd.DataFrame({'response':responselist, 'RMSE_base_test':RMSE_base_test, 'RMSE_base_new':RMSE_base_new})

#####################################################################################
# UNTUNED MODELS
ratelist = ['Citations', 'Altmetric Score', 'Patent Mentions', 'Policy Mentions']
model_typelist = ['Base Model', 'NLP Model', 'Uniqueness Model']
for rate in ratelist:
    for model_type in model_typelist:

        if model_type == 'Base Model':
            predictor = Xa
        elif model_type == 'NLP Model':
            predictor = Xb
        elif model_type == 'Uniqueness Model':
            predictor = Xc
            
        if rate == 'Citations':
            response = 'log_cpy_residual'
        elif rate == 'Altmetric Score':
            response = 'log_alt_residual'
        elif rate == 'Patent Mentions':
            response = 'log_patent_mentions_residual'
        elif rate == 'Policy Mentions':
            response = 'log_policy_mentions_residual'
            
        X_new, y_new, X_train, X_validation, y_train, y_validation, X_test, y_test = split_data(predictor, response)
        model, r2 = regression_model()
        regression_model_output(model, f'{rate}_{model_type}_untuned', rate, model_type)
        
#####################################################################################
# TUNING 

rate = _____ # 'Citations', 'Altmetric Score', 'Patent Mentions', 'Policy Mentions'
model_type = _____ # 'Base Model', 'NLP Model', 'Uniqueness Model'

if model_type == 'Base Model':
    predictor = Xa
elif model_type == 'NLP Model':
    predictor = Xb
elif model_type == 'Uniqueness Model':
    predictor = Xc
    
if rate == 'Citations':
    response = 'log_cpy_residual'
elif rate == 'Altmetric Score':
    response = 'log_alt_residual'
elif rate == 'Patent Mentions':
    response = 'log_patent_mentions_residual'
elif rate == 'Policy Mentions':
    response = 'log_policy_mentions_residual'
    
X_new, y_new, X_train, X_validation, y_train, y_validation, X_test, y_test = split_data(predictor, response)

model, best_params, r2 = regression_tuning({'learning_rate':[0.01, 0.05, 0.1, 0.3]}) 
learning_rate = best_params['learning_rate']

model, best_params, r2 = regression_tuning({'learning_rate':[learning_rate], 'max_depth': [3, 6, 10], 'min_child_weight': [1, 5, 10]}) 
max_depth = best_params['max_depth']
min_child_weight = best_params['min_child_weight']

if min_child_weight == 10:
    model, best_params, r2 = regression_tuning({'learning_rate':[learning_rate], 'max_depth': [max_depth], 'min_child_weight': [10, 15, 20]}) 
    min_child_weight = best_params['min_child_weight']

if min_child_weight == 20:
    model, best_params, r2 = regression_tuning({'learning_rate':[learning_rate], 'max_depth': [max_depth], 'min_child_weight': [20, 25, 30]}) 
    min_child_weight = best_params['min_child_weight']

if min_child_weight == 30:
    model, best_params, r2 = regression_tuning({'learning_rate':[learning_rate], 'max_depth': [max_depth], 'min_child_weight': [30, 35, 40]}) 
    min_child_weight = best_params['min_child_weight']

if min_child_weight == 40:
    model, best_params, r2 = regression_tuning({'learning_rate':[learning_rate], 'max_depth': [max_depth], 'min_child_weight': [40, 45, 50]}) 
    min_child_weight = best_params['min_child_weight']
    
model, best_params, r2 = regression_tuning({'learning_rate':[learning_rate], 'max_depth': [max_depth], 'min_child_weight': [min_child_weight], 'colsample_bytree': [0.6, 0.8, 1], 'subsample': [0.6, 0.8, 1]})
colsample_bytree = best_params['colsample_bytree']
subsample = best_params['subsample']

model, best_params, r2 = regression_tuning({'learning_rate':[learning_rate], 'max_depth': [max_depth], 'min_child_weight': [min_child_weight], 'colsample_bytree': [colsample_bytree], 'subsample': [subsample], 'gamma':[0, 0.1, 0.2, 0.3]}) 
g = best_params['gamma']

model, best_params, r2 = regression_tuning({'learning_rate':[learning_rate], 'max_depth': [max_depth], 'min_child_weight': [min_child_weight], 'colsample_bytree': [colsample_bytree], 'subsample': [subsample], 'gamma':[g], 'alpha': [0, 1, 2], 'lambda': [0, 1, 2]}) 
a = best_params['alpha']
l = best_params['lambda']

print(f'r2 = {r2} \nlearning_rate:{learning_rate} \nmax_depth:{max_depth} \nmin_child_weight:{min_child_weight} \ncolsample_bytree:{colsample_bytree} \nsubsample:{subsample} \ngamma:{g} \nalpha:{a} \nlambda:{l}')
joblib.dump(model, f'4_Model_Fitting/{model_type}_{rate} tuned.pkl') # Save the model to a file

#####################################################################################
# TUNED MODELS

ratelist = ['Citations', 'Patent Mentions', 'Altmetric Score', 'Policy Mentions']
model_typelist = ['Baseline Model', 'NLP Model', 'Uniqueness Model']

for model_type in model_typelist:
    for rate in ratelist:

        if model_type == 'Baseline Model':
            predictor = Xa
        elif model_type == 'NLP Model':
            predictor = Xb
        elif model_type == 'Uniqueness Model':
            predictor = Xc
            
        if rate == 'Citations':
            response = 'log_cpy_residual'
        elif rate == 'Altmetric Score':
            response = 'log_alt_residual'
        elif rate == 'Patent Mentions':
            response = 'log_patent_mentions_residual'
        elif rate == 'Policy Mentions':
            response = 'log_policy_mentions_residual'
            
        X_new, y_new, X_train, X_validation, y_train, y_validation, X_test, y_test = split_data(predictor, response)
        model = joblib.load(f'4_Model_Fitting/{model_type}_{rate} tuned.pkl')
        regression_model_output(model, f'{rate}_{model_type}_tuned', rate, model_type)
