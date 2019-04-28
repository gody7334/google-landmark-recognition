import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from pytorch_pretrained_bert.modeling import BertModel

class Head(nn.Module):
    """The MLP submodule"""
    def __init__(self, bert_hidden_size: int):
        super().__init__()
        self.bert_hidden_size = bert_hidden_size
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bert_hidden_size * 3),
            nn.Dropout(0.5),
            nn.Linear(bert_hidden_size * 3, bert_hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(bert_hidden_size),
            nn.Dropout(0.5),
            nn.Linear(bert_hidden_size, bert_hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(bert_hidden_size),
            nn.Dropout(0.5),
            nn.Linear(bert_hidden_size, 3)
        )
        for i, module in enumerate(self.fc):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                print("Initing batchnorm")
            elif isinstance(module, nn.Linear):
                if getattr(module, "weight_v", None) is not None:
                    nn.init.uniform_(module.weight_g, 0, 1)
                    nn.init.kaiming_normal_(module.weight_v)
                    print("Initing linear with weight normalization")
                    assert model[i].weight_g is not None
                else:
                    nn.init.kaiming_normal_(module.weight)
                    print("Initing linear")
                nn.init.constant_(module.bias, 0)

    def forward(self, bert_outputs, offsets):
        assert bert_outputs.size(2) == self.bert_hidden_size
        extracted_outputs = bert_outputs.gather(
            1, offsets.unsqueeze(2).expand(-1, -1, bert_outputs.size(2))
        ).view(bert_outputs.size(0), -1)
        return self.fc(extracted_outputs)


class GAPModel(nn.Module):
    """The main model."""
    def __init__(self, bert_model: str, device: torch.device):
        super().__init__()
        self.device = device
        if bert_model in ("bert-base-uncased", "bert-base-cased"):
            self.bert_hidden_size = 768
        elif bert_model in ("bert-large-uncased", "bert-large-cased"):
            self.bert_hidden_size = 1024
        else:
            raise ValueError("Unsupported BERT model.")
        self.bert = BertModel.from_pretrained(bert_model).to(device)
        self.head = Head(self.bert_hidden_size).to(device)

    def forward(self, token_tensor, offsets):
        token_tensor = token_tensor.to(self.device)
        bert_outputs, _ =  self.bert(
            token_tensor, attention_mask=(token_tensor > 0).long(),
            token_type_ids=None, output_all_encoded_layers=False)
        head_outputs = self.head(bert_outputs, offsets.to(self.device))
        return head_outputs

    def set_dropout_prob(self, m, prob=0.0):
        '''
        m: node start to search
        '''
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())

        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.modules.dropout._DropoutNd):
                m.p = prob
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)

        apply_leaf(m, prob)


class GAPModel_CheckPoint(GAPModel):
    '''
    make bert modules as check point module,
    train will run twice on this module
    remove dropout ratio (prob=0.0), otherwise,
    second run will change dropout node, different from frist run nodes,
    so gradient from first run will be wrong
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_dropout_prob(self.bert, prob=0.0)

    def check_point(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0], attention_mask=(inputs[0]>0).long(),
                    token_type_ids=None, output_all_encoded_layers=False)
            return inputs
        return custom_forward

    def forward(self, token_tensor, offsets):
        token_tensor = token_tensor.to(self.device)
        bert_outputs, _ = checkpoint.checkpoint(self.check_point(self.bert), token_tensor)
        head_outputs = self.head(bert_outputs, offsets.to(self.device))
        return head_outputs


