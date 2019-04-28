import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

class score(torch.nn.Module):

    def __init__(self, embed_dim, hidden_dim):
        super(score, self).__init__()
        self.score = torch.nn.Sequential(
                     torch.nn.Linear(embed_dim, hidden_dim),
                     torch.nn.ReLU(inplace=True),
                     torch.nn.Dropout(0.6),
                     torch.nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.score(x)

class mentionpair_score(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(mentionpair_score, self).__init__()
        self.score = score(input_dim, hidden_dim)

    def forward(self, g1, g2, dist_embed):

        element_wise = g1 * g2
        pair_score   = self.score(torch.cat((g1, g2, element_wise, dist_embed), dim=-1))

        return pair_score

class score_model(torch.nn.Module):

    def __init__(self, bert_model=''):
        super(score_model, self).__init__()

        if bert_model in ("bert-base-uncased", "bert-base-cased"):
            self.bert_hidden_size = 768
        elif bert_model in ("bert-large-uncased", "bert-large-cased"):
            self.bert_hidden_size = 1024
        else:
            raise ValueError("Unsupported BERT model.")

        self.buckets_embedding_size = 20
        self.score_hidden_size = 128

        self.buckets        = [1, 2, 3, 4, 5, 8, 16, 32, 64]
        self.bert           = BertModel.from_pretrained(bert_model)
        self.embedding      = torch.nn.Embedding(len(self.buckets)+1,
                                self.buckets_embedding_size)
        self.span_extractor = EndpointSpanExtractor(self.bert_hidden_size, "x,y,x*y")
        self.pair_score     = mentionpair_score(self.bert_hidden_size*3*3 \
                                + self.buckets_embedding_size,
                                self.score_hidden_size)

    def forward(self, sent, offsets, distP_A, distP_B):

        bert_output, _   = self.bert(sent, output_all_encoded_layers=False) # (batch_size, max_len, 768)
        #Distance Embeddings
        distPA_embed     = self.embedding(distP_A)
        distPB_embed     = self.embedding(distP_B)

        #Span Representation
        span_repres     = self.span_extractor(bert_output, offsets) #(batch, 3, 2304)
        span_repres     = torch.unbind(span_repres, dim=1) #[A: (bath, 2304), B: (bath, 2304), Pronoun:  (bath, 2304)]
        span_norm = []
        for i in range(len(span_repres)):
            span_norm.append(F.normalize(span_repres[i], p=2, dim=1)) #normalizes the words embeddings

        ap_score = self.pair_score(span_norm[2], span_norm[0], distPA_embed)
        bp_score = self.pair_score(span_norm[2], span_norm[1], distPB_embed)

        # loss is calculated using "Softmax", if nether a or b,
        # ap_score and bp_score will be negative, nan_score 0 will be the biggest value
        nan_score = torch.zeros_like(ap_score)
        output = torch.cat((ap_score, bp_score, nan_score), dim=1)

        return output

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


class Head(nn.Module):
    """The MLP submodule"""
    def __init__(self, bert_hidden_size: int):
        super().__init__()
        self.bert_hidden_size = bert_hidden_size
        fc_size = 256
        # self.span_extractor = SelfAttentiveSpanExtractor(bert_hidden_size)
        self.span_extractor = EndpointSpanExtractor(
            bert_hidden_size, "x,y,x*y"
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bert_hidden_size * 7),
            nn.Dropout(0.5),
            nn.Linear(bert_hidden_size * 7, fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc_size),
            nn.Dropout(0.5),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc_size),
            nn.Dropout(0.5),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc_size),
            nn.Dropout(0.5),
            nn.Linear(fc_size, 3)
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
        spans_contexts = self.span_extractor(
            bert_outputs,
            offsets[:, :4].reshape(-1, 2, 2)
        ).reshape(offsets.size()[0], -1)
        return self.fc(torch.cat([
            spans_contexts,
            torch.gather(
                bert_outputs, 1,
                offsets[:, [4]].unsqueeze(2).expand(-1, -1, self.bert_hidden_size)
            ).squeeze(1)
        ], dim=1))

class GAPModel(nn.Module):
    """The main model."""
    def __init__(self, bert_model: str, device: torch.device, use_layer: int = -2):
        super().__init__()
        self.device = device
        self.use_layer = use_layer
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
            token_type_ids=None, output_all_encoded_layers=True)
        head_outputs = self.head(bert_outputs[self.use_layer], offsets.to(self.device))
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


