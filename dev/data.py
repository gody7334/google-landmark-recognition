import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
from utils.project import Global as G
from spacy.lang.en import English
from tqdm import tqdm
from ast import literal_eval

BERT_MODEL = 'bert-base-uncased'
CASED = False

G.logger.info("BERT_MODEL %s", BERT_MODEL)
G.logger.info("CASED %s", str(CASED))

def prepare_dist_df(df):
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

    return extract_dist_features(df)

def prepare_token_df(df, tokenizer):
    nlp = English()
    sentencizer = nlp.create_pipe('sentencizer')
    nlp.add_pipe(sentencizer)

    def candidate_length(candidate):
        #count the word length without space
        count = 0
        for i in range(len(candidate)):
            if candidate[i] !=  " ": count += 1
        return count

    def count_char(text, offset):
        count = 0
        for pos in range(offset):
            if text[pos] != " ": count +=1
        return count

    def count_token_length_special(token):
        count = 0
        special_token = ["#", " "]
        for i in range(len(token)):
            if token[i] not in special_token:
                count+=1
        return count

    def find_word_index(tokenized_text, char_start, target):
        tar_len = candidate_length(target)
        char_count = 0
        word_index = []
        special_token = ["[CLS]", "[SEP]"]
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if char_count in range(char_start, char_start+tar_len):
                if token in special_token: # for the case like "[SEP]. she"
                    continue
                word_index.append(i)
            if token not in special_token:
                token_length = count_token_length_special(token)
                char_count += token_length

        if len(word_index) == 1:
            return [word_index[0], word_index[0]] #the output will be start index of span, and end index of span
        else:
            return [word_index[0], word_index[-1]]

    def create_tokenizer_input(sents):
        tokenizer_input = str()
        for i, sent in enumerate(sents):
            if i == 0:
                tokenizer_input += "[CLS] "+sent.text+" [SEP] "
            elif i == len(sents) - 1:
                tokenizer_input += sent.text+" [SEP]"
            else:
                tokenizer_input += sent.text+" [SEP] "

        return  tokenizer_input

    def create_inputs(dataframe):
        idxs = dataframe.index
        columns = ['indexed_token', 'offset']
        features_df = pd.DataFrame(index=idxs, columns=columns)
        max_len = 0
        for i in tqdm(range(len(dataframe))):
            text           = dataframe.loc[i, 'Text']
            Pronoun_offset = dataframe.loc[i, 'Pronoun-offset']
            A_offset       = dataframe.loc[i, "A-offset"]
            B_offset       = dataframe.loc[i, "B-offset"]
            Pronoun        = dataframe.loc[i, "Pronoun"]
            A              = dataframe.loc[i, "A"]
            B              = dataframe.loc[i, "B"]
            doc            = nlp(text)

            sents = []
            for sent in doc.sents: sents.append(sent)
            token_input = create_tokenizer_input(sents)
            token_input = token_input.replace("#", "*") #Remove special symbols “#” from the original sentence
            tokenized_text = tokenizer.tokenize(token_input) #the token text
            if len(tokenized_text) > max_len:
                max_len = len(tokenized_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) #token text to index

            A_char_start, B_char_start = count_char(text, A_offset), count_char(text, B_offset)
            Pronoun_char_start         = count_char(text, Pronoun_offset)

            word_indexes = [] #
            for char_start, target in zip([A_char_start, B_char_start, Pronoun_char_start], [A, B, Pronoun]):
                word_indexes.append(find_word_index(tokenized_text, char_start, target))#
            features_df.iloc[i] = [indexed_tokens, word_indexes]

        print('max length of sentence:', max_len)
        return features_df

    df = create_inputs(df)
    return df

def extract_target(df):
    df["Neither"] = 0
    df.loc[~(df['A-coref'] | df['B-coref']), "Neither"] = 1
    df["target"] = 0
    df.loc[df['B-coref'] == 1, "target"] = 1
    df.loc[df["Neither"] == 1, "target"] = 2
    print(df.target.value_counts())
    return df

def collate_examples(batch, truncate_len=500):
    """Batch preparation.

    1. Pad the sequences
    2. Transform the target.
    index_token, offset, distP_A, distP_B, label
    """

    transposed = list(zip(*batch))

    max_len = min(
        max((len(x) for x in transposed[0])),
        truncate_len
    )
    tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    for i, row in enumerate(transposed[0]):
        row = np.array(row[:truncate_len])
        tokens[i, :len(row)] = row
    token_tensor = torch.from_numpy(tokens)

    # Offsets
    offsets = torch.stack([
        torch.LongTensor(x) for x in transposed[1]
    ], dim=0) # Account for the [CLS] token

    # distP_A
    distP_A = torch.LongTensor(transposed[2])

    # distP_B
    distP_B = torch.LongTensor(transposed[3])

    # Labels
    if len(transposed) == 4:
        return token_tensor, offsets, distP_A, distP_B, None
    labels = torch.LongTensor(transposed[4])
    return token_tensor, offsets, distP_A, distP_B, labels

class GAPDataset(Dataset):

    def __init__(self, dataframe, tokenizer, transform=None, labeled=True):
        self.df = dataframe
        self.transform = transform
        self.tokenizer = tokenizer
        self.labeled = labeled

        token_df = prepare_token_df(self.df, self.tokenizer)
        dist_df = prepare_dist_df(self.df)

        if labeled:
            self.df = extract_target(pd.concat([self.df, token_df, dist_df], axis=1, sort=False))
            self.y = self.df.target.values.astype("uint8")
        else:
            self.df = pd.concat([self.df, token_df, dist_df], axis=1, sort=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        index_token = self.df.loc[idx, 'indexed_token']
        # index_token = literal_eval(index_token) # Change string to list
        index_token = pad_sequences([index_token], maxlen=360, padding='post')[0] #pad

        offset = self.df.loc[idx, 'offset']
        # offset = literal_eval(offset)
        offset = np.asarray(offset, dtype='int32')
        if self.labeled: label  = int(self.df.loc[idx, 'target']);

        distP_A = self.df.loc[idx, 'D_PA']
        distP_B = self.df.loc[idx, 'D_PB']

        if self.transform:
            index_token = self.transform(index_token)
            offset = self.transform(offset)
            if self.labeled: label = self.transform(label);

        if self.labeled:
            return index_token, offset, distP_A, distP_B, label
        return index_token, offset, distP_A, distP_B

class GAPDataLoader():
    def __init__(self,
            df_train,
            df_val,
            df_test,
            sample_sub,
            train_size=20,
            val_size=128,
            test_size=128):
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.sample_sub = sample_sub
        self.df_submission = None
        # df_train = pd.read_csv("~/gender-pronoun/input/dataset/gap-test.csv")
        # df_val = pd.read_csv("~/gender-pronoun/input/dataset/gap-validation.csv")
        # df_test = pd.read_csv("~/gender-pronoun/input/dataset/gap-development.csv")
        # sample_sub = pd.read_csv("~/gender-pronoun/input/dataset/sample_submission_stage_1.csv")
        # assert sample_sub.shape[0] == df_test.shape[0]

        self.tokenizer = None
        self.bert_tokenizer()
        self.train_ds = GAPDataset(self.df_train, self.tokenizer)
        self.val_ds = GAPDataset(self.df_val, self.tokenizer)
        self.test_ds = GAPDataset(self.df_test, self.tokenizer,labeled=True)
        self.submission_ds = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.submission_loader = None

        self.update_batch_size(train_size,val_size,test_size)

    def set_submission_dataloader(self, df_submission):
        self.df_submission = df_submission
        self.submission_ds = GAPDataset(self.df_submission, self.tokenizer,labeled=False)
        self.submission_loader = DataLoader(
            self.submission_ds,
            collate_fn = collate_examples,
            batch_size=self.test_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False
        )

    def update_batch_size(self,
            train_size=20,
            val_size=128,
            test_size=128):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.get_dataloader()

    def bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokenizer = BertTokenizer.from_pretrained(
            # BERT_MODEL,
            # do_lower_case=CASED,
            # never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        # )
        self.tokenizer = tokenizer

    def get_dataloader(self):
        self.train_loader = DataLoader(
            self.train_ds,
            collate_fn = collate_examples,
            batch_size=self.train_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_ds,
            collate_fn = collate_examples,
            batch_size=self.val_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_ds,
            collate_fn = collate_examples,
            batch_size=self.test_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False
        )


def ut_gap_dataloader():

    def sample_data_from_ds(ds):
        num = len(ds)
        for m in range(num):
            i = np.random.choice(num)
            row = ds.df.loc[i]
            # text = insert_tag(row)
            (index_token, offset, distP_A, distP_B), label = ds[i]
            import ipdb; ipdb.set_trace();

    dl=GAPDataLoader()
    # sample_data_from_ds(dl.train_ds)
    sample_data_from_ds(dl.train_ds)
    # sample_data_from_ds(dl.test_ds)

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    ut_gap_dataloader()
    print('success!')
