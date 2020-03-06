#!/usr/bin/env python3

import pandas as pd
import numpy as np 

data_columns = ['fixed_acidity', 
                'volatile_acidity', 
                'citric_acid', 
                'residual_sugar',
                'chlorides',
                'free_sulfur_dioxide',
                'total_sulfur_dioxide',
                'density',
                'pH',
                'sulphates',
                'alcohol',
                'quality', # ordinal (0 to 10)
                'wine_type'] # categorical (2 categories)

df_data = pd.read_csv('./data.csv', header=None)

df_data.columns = data_columns

df_data.to_csv('./wine.csv', index=False)


