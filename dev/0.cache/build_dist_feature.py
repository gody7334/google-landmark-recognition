# code is referenced from
#https://www.kaggle.com/keyit92/coref-by-mlp-cnn-coattention

import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from spacy.lang.en import English

nlp = English()

def bs(lens, target):
    low, high = 0, len(lens) - 1

    while low < high:
        mid = low + int((high - low) / 2)

        if target > lens[mid]:
            low = mid + 1
        elif target < lens[mid]:
            high = mid
        else:
            return mid + 1

    return low

def bin_distance(dist):

    buckets = [1, 2, 3, 4, 5, 8, 16, 32, 64]
    low, high = 0, len(buckets)
    while low < high:
        mid = low + int((high-low) / 2)
        if dist > buckets[mid]:
            low = mid + 1
        elif dist < buckets[mid]:
            high = mid
        else:
            return mid

    return low

def distance_features(P, A, B, char_offsetP, char_offsetA, char_offsetB, text, URL):

    doc = nlp(text)

    lens = [token.idx for token in doc]
    mention_offsetP = bs(lens, char_offsetP) - 1
    mention_offsetA = bs(lens, char_offsetA) - 1
    mention_offsetB = bs(lens, char_offsetB) - 1

    mention_distA = mention_offsetP - mention_offsetA
    mention_distB = mention_offsetP - mention_offsetB

    splited_A = A.split()[0].replace("*", "")
    splited_B = B.split()[0].replace("*", "")

    if re.search(splited_A[0], str(URL)):
        contains = 0
    elif re.search(splited_B[0], str(URL)):
        contains = 1
    else:
        contains = 2

    dist_binA = bin_distance(mention_distA)
    dist_binB = bin_distance(mention_distB)
    output =  [dist_binA, dist_binB, contains]

    return output

def extract_dist_features(df):

    index = df.index
    columns = ["D_PA", "D_PB", "IN_URL"]
    dist_df = pd.DataFrame(index = index, columns = columns)

    for i in tqdm(range(len(df))):

        text = df.loc[i, 'Text']
        P_offset = df.loc[i,'Pronoun-offset']
        A_offset = df.loc[i, 'A-offset']
        B_offset = df.loc[i, 'B-offset']
        P, A, B  = df.loc[i,'Pronoun'], df.loc[i, 'A'], df.loc[i, 'B']
        URL = df.loc[i, 'URL']

        dist_df.iloc[i] = distance_features(P, A, B, P_offset, A_offset, B_offset, text, URL)

    return dist_df

train_df = pd.read_csv("~/gender-pronoun/input/dataset/gap-test.csv")
val_df = pd.read_csv("~/gender-pronoun/input/dataset/gap-validation.csv")
test_df = pd.read_csv("~/gender-pronoun/input/dataset/gap-development.csv")

test_dist_df = extract_dist_features(test_df)
test_dist_df.to_csv('~/gender-pronoun/input/dataset/dist_df_test.csv', index=False)
val_dist_df = extract_dist_features(val_df)
val_dist_df.to_csv('~/gender-pronoun/input/dataset/dist_df_val.csv', index=False)
train_dist_df = extract_dist_features(train_df)
train_dist_df.to_csv('~/gender-pronoun/input/dataset/dist_df_train.csv', index=False)
