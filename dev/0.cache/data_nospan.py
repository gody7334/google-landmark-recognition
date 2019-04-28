import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from utils.project import Global as G

BERT_MODEL = 'bert-large-uncased'
CASED = False

G.logger.info("BERT_MODEL %s", BERT_MODEL)
G.logger.info("CASED %s", str(CASED))

df_train = pd.read_csv("~/gender-pronoun/input/gap-test.tsv", delimiter="\t")
df_val = pd.read_csv("~/gender-pronoun/input/gap-validation.tsv", delimiter="\t")
df_test = pd.read_csv("~/gender-pronoun/input/gap-development.tsv", delimiter="\t")
sample_sub = pd.read_csv("~/gender-pronoun/input/sample_submission_stage_1.csv")
assert sample_sub.shape[0] == df_test.shape[0]

def insert_tag(row):
    """Insert custom tags to help us find the position of A, B, and the pronoun after tokenization."""
    to_be_inserted = sorted([
        (row["A-offset"], " [A] "),
        (row["B-offset"], " [B] "),
        (row["Pronoun-offset"], " [P] ")
    ], key=lambda x: x[0], reverse=True)
    text = row["Text"]
    for offset, tag in to_be_inserted:
        text = text[:offset] + tag + text[offset:]
    return text

def tokenize(text, tokenizer):
    """Returns a list of tokens and the positions of A, B, and the pronoun."""
    entries = {}
    final_tokens = []
    for token in tokenizer.tokenize(text):
        if token in ("[A]", "[B]", "[P]"):
            entries[token] = len(final_tokens)
            continue
        final_tokens.append(token)
    return final_tokens, (entries["[A]"], entries["[B]"], entries["[P]"])

def collate_examples(batch, truncate_len=500):
    """Batch preparation.

    1. Pad the sequences
    2. Transform the target.
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
    ], dim=0) + 1 # Account for the [CLS] token
    # Labels
    if len(transposed) == 2:
        return token_tensor, offsets, None
    one_hot_labels = torch.stack([
        torch.from_numpy(x.astype("uint8")) for x in transposed[2]
    ], dim=0)
    _, labels = one_hot_labels.max(dim=1)
    return token_tensor, offsets, labels

class GAPDataset(Dataset):
    """Custom GAP Dataset class"""
    def __init__(self, df, tokenizer, labeled=True):
        self.labeled = labeled
        self.df = df
        if labeled:
            tmp = df[["A-coref", "B-coref"]].copy()
            tmp["Neither"] = ~(df["A-coref"] | df["B-coref"])
            self.y = tmp.values.astype("bool")
        # Extracts the tokens and offsets(positions of A, B, and P)
        self.offsets, self.tokens = [], []
        for _, row in df.iterrows():
            text = insert_tag(row)
            tokens, offsets = tokenize(text, tokenizer)
            self.offsets.append(offsets)
            self.tokens.append(tokenizer.convert_tokens_to_ids(
                ["[CLS]"] + tokens + ["[SEP]"]))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labeled:
            return self.tokens[idx], self.offsets[idx], self.y[idx]
        return self.tokens[idx], self.offsets[idx], None

class GAPDataLoader():
    def __init__(self,
            train_size=20,
            val_size=128,
            test_size=128):
        self.tokenizer = None
        self.bert_tokenizer()
        self.train_ds = GAPDataset(df_train, self.tokenizer)
        self.val_ds = GAPDataset(df_val, self.tokenizer)
        self.test_ds = GAPDataset(df_test, self.tokenizer,labeled=True)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.update_batch_size(train_size,val_size,test_size)

    def update_batch_size(self,
            train_size=20,
            val_size=128,
            test_size=128):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.get_dataloader()

    def bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(
            BERT_MODEL,
            do_lower_case=CASED,
            never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[A]", "[B]", "[P]")
        )
        # These tokens are not actually used, so we can assign arbitrary values.
        tokenizer.vocab["[A]"] = -1
        tokenizer.vocab["[B]"] = -1
        tokenizer.vocab["[P]"] = -1
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
            text = insert_tag(row)
            tokens, offsets, y = ds[i]
            import ipdb; ipdb.set_trace();

    dl=GAPDataLoader()
    # sample_data_from_ds(dl.train_ds)
    sample_data_from_ds(dl.val_ds)
    # sample_data_from_ds(dl.test_ds)

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    ut_gap_dataloader()
    print('success!')
