import os
import time
import pandas as pd


df_train = pd.read_csv("~/gender-pronoun/input/dataset/gap-development.tsv", delimiter="\t")
df_val = pd.read_csv("~/gender-pronoun/input/dataset/gap-validation.tsv", delimiter="\t")
df_test = pd.read_csv("~/gender-pronoun/input/dataset/gap-test.tsv", delimiter="\t")
sample_sub = pd.read_csv("~/gender-pronoun/input/dataset/sample_submission_stage_1.csv")

df_correct = pd.read_csv("~/gender-pronoun/input/dataset/corrections.csv")

df_c_train = df_correct[df_correct['ID'].str.contains('development')]
df_c_val = df_correct[df_correct['ID'].str.contains('validation')]
df_c_test = df_correct[df_correct['ID'].str.contains('test')]

df_c_train = pd.merge(df_train, df_c_train, on="ID", how="outer")
df_c_train.loc[df_c_train['Correction']=='A','A-coref'] = True
df_c_train.loc[df_c_train['Correction']=='A','B-coref'] = False
df_c_train.loc[df_c_train['Correction']=='B','B-coref'] = True
df_c_train.loc[df_c_train['Correction']=='B','A-coref'] = False
df_c_train.loc[df_c_train['Correction']=='N','A-coref'] = False
df_c_train.loc[df_c_train['Correction']=='N','B-coref'] = False

df_c_val = pd.merge(df_val, df_c_val, on="ID", how="outer")
df_c_val.loc[df_c_val['Correction']=='A','A-coref'] = True
df_c_val.loc[df_c_val['Correction']=='A','B-coref'] = False
df_c_val.loc[df_c_val['Correction']=='B','B-coref'] = True
df_c_val.loc[df_c_val['Correction']=='B','A-coref'] = False
df_c_val.loc[df_c_val['Correction']=='N','A-coref'] = False
df_c_val.loc[df_c_val['Correction']=='N','B-coref'] = False

df_c_test = pd.merge(df_test, df_c_test, on="ID", how="outer")
df_c_test.loc[df_c_test['Correction']=='A','A-coref'] = True
df_c_test.loc[df_c_test['Correction']=='A','B-coref'] = False
df_c_test.loc[df_c_test['Correction']=='B','B-coref'] = True
df_c_test.loc[df_c_test['Correction']=='B','A-coref'] = False
df_c_test.loc[df_c_test['Correction']=='N','A-coref'] = False
df_c_test.loc[df_c_test['Correction']=='N','B-coref'] = False

df_c_train.to_csv("~/gender-pronoun/input/dataset/correct-gap-development.csv")
df_c_val.to_csv("~/gender-pronoun/input/dataset/correct-gap-validation.csv")
df_c_test.to_csv("~/gender-pronoun/input/dataset/correct-gap-test.csv")

