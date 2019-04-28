import os
import time
import pandas as pd


def correct_df(df):
    df['Text_bt'] = df['Text_bt'].str.replace('<*\s*<\s*\[*\s*\[\s*A\s*\]*\s*\]\s*>*\s*>',"<< [[A]] >>",regex=True)
    df['Text_bt'] = df['Text_bt'].str.replace('<*\s*<\s*\[*\s*\[\s*B\s*\]*\s*\]\s*>*\s*>',"<< [[B]] >>",regex=True)
    df['Text_bt'] = df['Text_bt'].str.replace('<*\s*<\s*\[*\s*\[\s*P\s*\]*\s*\]\s*>*\s*>',"<< [[P]] >>",regex=True)

    df['Text_bt'] = df['Text_bt'].str.replace('<*\s*<*\s*\[\[A\]*\]\s*>*\s*>*'," << [[A]] >> ",regex=True)
    df['Text_bt'] = df['Text_bt'].str.replace('<*\s*<*\s*\[\[B\]*\]\s*>*\s*>*'," << [[B]] >> ",regex=True)
    df['Text_bt'] = df['Text_bt'].str.replace('<*\s*<*\s*\[\[P\]*\]\s*>*\s*>*'," << [[P]] >> ",regex=True)

    df_correct = df[df['Text_bt'].str.contains('(?=.*<< \[\[A\]\] >>)(?=.*<< \[\[B\]\] >>)(?=.*<< \[\[P\]\] >>)',regex=True)]
    df_incoorect = df[~df.index.isin(df_correct.index)]

    df_correct['A-offset'] = df_correct['Text_bt'].str.find('<< [[A]] >>')
    df_correct['B-offset'] = df_correct['Text_bt'].str.find('<< [[B]] >>')
    df_correct['Pronoun-offset'] = df_correct['Text_bt'].str.find('<< [[P]] >>')

    import ipdb; ipdb.set_trace();

df_train = pd.read_csv("~/gender-pronoun/input/dataset/trans-gap-development.csv")
df_val = pd.read_csv("~/gender-pronoun/input/dataset/trans-gap-validation.csv")
df_test = pd.read_csv("~/gender-pronoun/input/dataset/trans-gap-test.csv")
sample_sub = pd.read_csv("~/gender-pronoun/input/dataset/sample_submission_stage_1.csv")

correct_df(df_train)

import ipdb; ipdb.set_trace();
