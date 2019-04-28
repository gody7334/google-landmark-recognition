import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import *

class BASE_EDA:
    def __init__(self, train_path='', test_path=''):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None

        if self.train_path is not '':
            self.train_df = self.load_df(self.train_path)
        if self.test_path is not '':
            self.test_df = self.load_df(self.test_path)

    def load_df(self, path):
        df = pd.read_csv(path)
        print('\nDataframe:')
        print(df.head())
        print("\nDataframe size", df.shape)
        return df

    def print_head(self, df,n=10):
        print('\nHead:')
        print(df.head(n))

    def label_freq(self, df, column):
        freq = pd.DataFrame(df.landmark_id.value_counts())
        freq.reset_index(inplace=True)
        freq.columns = [column,'count']
        freq.sort_values(by=['count'])
        print('\nFrequency:')
        print(freq.head())
        return freq

    def missing_value(self, df):
        total = df.isnull().sum().sort_values(ascending = False)
        percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending = False)
        missing_train_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        print('\nMissing value sum:')
        print(missing_train_data.head())

    def print_unique(self, df):
        unique = df.nunique()
        unique.columns = ['column','count']
        print('\nUnique value:')
        print(unique)

    def less_freq_label(self, df, column, less_than=10):
        less_freq_df = df[df[column]<less_than]
        print(f'\nless freq label %, less than {less_than} labels:')
        print(len(less_freq_df)/len(df))

    # def displot()

    def bar_chart(self, df, x, y, title='barchart'):
        plt.figure()
        plt.title(title)
        sns.set_color_codes("pastel")
        sns.barplot(x=x, y=y, data=df, label="Count")
        plt.show()
        savefig('./cache/'+title+'.png')
        clf()

class LANDMARK_EDA(BASE_EDA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_label_freq = None

    def do_eda(self):
        self.train_label_freq = self.label_freq(self.train_df, 'landmark_id')
        import ipdb; ipdb.set_trace();

        for i in range(2,20):
            self.less_freq_label(self.train_label_freq, 'count', i)

        self.missing_value(self.train_df)
        self.print_unique(self.train_df)

        train_label_top_freq = self.train_label_freq.head(n=100)
        self.bar_chart(train_label_top_freq, 'landmark_id', 'count', 'landmark_freq')

    def reduce_df(self):
        df = self.train_df[['id','landmark_id']]
        df.to_csv('/home/gody7334/google-landmark/input/dataset/train_exp.csv')
        import ipdb; ipdb.set_trace();




if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    train_path = '/home/gody7334/google-landmark/input/train.csv'
    test_path = ''
    eda = LANDMARK_EDA(train_path, test_path)
    eda.do_eda()
    print('success!')
