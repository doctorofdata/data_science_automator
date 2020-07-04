# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import sys
import zfuncs
import ast
import os

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

plt.style.use('fivethirtyeight')

# Load data
data = pd.read_csv(input('Enter path of desired data file: '))

# Create output directory
outdir = input('Enter the path for the desired output directory: ')

try:
    
    os.mkdir(outdir)
    
except:
    
    print('Directory exists..')
    pass

# Label encode the categorical features
cats = input('Enter list categorical variables for transformation: ')

if cats:
    
    # Initialize encoder
    le = LabelEncoder()

    # Iterate
    for col in cats:
    
        data[col] = le.fit_transform(data[col])
        
# Get input for model variables
target_variable = input('Enter column name of desired Y variable: ')

x = data[[i for i in data.columns if i != target_variable]]

# Read input for modeling
with open('models.txt', 'rb') as f:
    
    models = f.read()
    
models = ast.literal_eval(models)

# Iterate user input to build
n = 0

for k, v in models.items():
    
    print('Running {} model..'.format(k))
    n += 1
    
    # Get predictions
    preds = zfuncs.build_dat_mod(v)
    
    # Evaluate
    zfuncs.model_evaluation(yval, preds, n)
    
    print('Training complete for {} model..'.format(k))
