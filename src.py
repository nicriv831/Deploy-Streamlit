
import sys
import os

# Get the path to the 'src' directory
src_path = os.path.abspath(os.path.join(os.getcwd(), '/src'))

# Add the 'src' folder to sys.path
if src_path not in sys.path:
  sys.path.append(src_path)

import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import FunctionTransformer

from sklearn import set_config
set_config(transform_output='pandas')

def replace_redund(df):
  df = df.copy()
  df = df.replace({'Student':'Unemployed', 'Businessman': 'Working', 'Maternity leave':'Working', 'Incomplete higher': 'Secondary',
                   'Secondary / secondary special': 'Secondary', 'Academic Degree': 'Higher education'})
  return df

redundant = FunctionTransformer(replace_redund)

def calc_cols(df):
  df = df.copy()
  monthly_income = df['AMT_INCOME_TOTAL'] / 12
  df['DTI'] = df['AMT_ANNUITY'] / monthly_income
  return df

add_cols = FunctionTransformer(calc_cols)

def drop_xna(df):
  df = df.copy()
  df= df[~df.isin(['XNA']).any(axis=1)]
  return df

xna_drop = FunctionTransformer(drop_xna)

def unk_drop(df):
  df=df.copy()
  df = df[~df.isin(['Unknown']).any(axis=1)]
  return df

drop_unks = FunctionTransformer(unk_drop)
