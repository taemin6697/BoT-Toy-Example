import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class FeedforwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        _, _, _, dim = q.size()
        k_t = k.transpose(-1, -2)
        score = torch.matmul(q, k_t) / math.sqrt(dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        score = self.softmax(score)
        return torch.matmul(score, v), score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.attn = Attention()

    def forward(self, q, k, v, mask=None):
        b, n, d = q.size()
        q, k, v = self.q(q), self.k(k), self.v(v)
        q = q.view(b, n, self.head, -1).transpose(1, 2)
        k = k.view(b, n, self.head, -1).transpose(1, 2)
        v = v.view(b, n, self.head, -1).transpose(1, 2)
        out, score = self.attn(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        out = self.o(out)

        return out, score

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, head):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, head)
        self.ffn = FeedforwardNetwork(d_model, d_model*2)

    def forward(self, x, mask=None):
        out, score = self.mha(x, x, x, mask)
        out = F.layer_norm(out + x, x.size()[1:])
        out = F.layer_norm(out + self.ffn(out), x.size()[1:])
        return out, score

class BoTEncoder(nn.Module):
    def __init__(self, d_model, head, layer_num=6, mode='Hard'):
        super().__init__()
        self.mode = mode
        self.layers = nn.ModuleList()

        for i in range(layer_num):
            if mode == 'Hard':
                self.layers.append(TransformerEncoderLayer(d_model, head))
            elif mode == 'Mix':
                if i % 2 == 0:
                    self.layers.append(TransformerEncoderLayer(d_model, head))
                else:
                    self.layers.append(TransformerEncoderLayer(d_model, head))

    def forward(self, x, mask=None):
        attention_maps = []
        for i, layer in enumerate(self.layers):
            if self.mode == 'Mix' and i % 2 == 1:
                x, _ = layer(x, mask=None)
            else:
                x, _ = layer(x, mask)
            attention_maps.append(_)
        return x, attention_maps


class BoT_Detokenizer(nn.Module):
    def __init__(self, embedding_dim, output_activation=None, device='cuda'):

        super(BoT_Detokenizer, self).__init__()
        self.device = device
        self.output_activation = output_activation

        # ModuleDict with explicitly defined names
        self.detokenizers = nn.ModuleDict({
            "forehand": nn.Linear(embedding_dim, 4),
            "wrist": nn.Linear(embedding_dim, 1),
            "palm": nn.Linear(embedding_dim, 1),
            "ffknuckle": nn.Linear(embedding_dim, 1),
            "ffproximal": nn.Linear(embedding_dim, 1),
            "ffmiddle": nn.Linear(embedding_dim, 1),
            "ffdistal": nn.Linear(embedding_dim, 1),
            "mfknuckle": nn.Linear(embedding_dim, 1),
            "mfproximal": nn.Linear(embedding_dim, 1),
            "mfmiddle": nn.Linear(embedding_dim, 1),
            "mfdistal": nn.Linear(embedding_dim, 1),
            "rfknuckle": nn.Linear(embedding_dim, 1),
            "rfproximal": nn.Linear(embedding_dim, 1),
            "rfmiddle": nn.Linear(embedding_dim, 1),
            "rfdistal": nn.Linear(embedding_dim, 1),
            "lfmetacarpal": nn.Linear(embedding_dim, 1),
            "lfknuckle": nn.Linear(embedding_dim, 1),
            "lfproximal": nn.Linear(embedding_dim, 1),
            "lfmiddle": nn.Linear(embedding_dim, 1),
            "lfdistal": nn.Linear(embedding_dim, 1),
            "thbase": nn.Linear(embedding_dim, 1),
            "thproximal": nn.Linear(embedding_dim, 1),
            "thhub": nn.Linear(embedding_dim, 1),
            "thmiddle": nn.Linear(embedding_dim, 1),
            "thdistal": nn.Linear(embedding_dim, 1)
        })

    def forward(self, x):
        batch_size = x.size(0)
        action_list = []

        # Explicitly decode actions based on predefined joint names
        for i, joint_name in enumerate(self.detokenizers.keys()):
            token_x = x[:, i, :]  # Extract token for current joint
            joint_action = self.detokenizers[joint_name](token_x)  # Decode using corresponding Linear layer
            action_list.append(joint_action)

        # Concatenate all joint actions into a single action vector
        action = torch.cat(action_list, dim=-1)
        return action



class BoT_Tokenizer(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model
        self.zero_token = torch.nn.Parameter(torch.zeros(1, self.d_model), requires_grad=False)

        self.tokenizers = nn.ModuleDict({
            "forehand": nn.Sequential(nn.Linear(in_features=9, out_features=d_model, bias=True)),
            "wrist": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "palm": nn.Sequential(nn.Linear(in_features=7, out_features=d_model, bias=True)),
            "ffknuckle": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "ffproximal": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "ffmiddle": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "ffdistal": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "mfknuckle": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "mfproximal": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "mfmiddle": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "mfdistal": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "rfknuckle": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "rfproximal": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "rfmiddle": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "rfdistal": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "lfmetacarpal": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "lfknuckle": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "lfproximal": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "lfmiddle": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "lfdistal": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "thbase": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "thproximal": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "thhub": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "thmiddle": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True)),
            "thdistal": nn.Sequential(nn.Linear(in_features=1, out_features=d_model, bias=True))
        })

        # Manual mapping for sensors
        self.sensor_map = {
            "forehand": [0, 1, 2, 27, 28, 32, 33, 34, 38],
            "wrist": [3],
            "palm": [4, 29, 30, 31, 35, 36, 37],
            "ffknuckle": [5],
            "ffproximal": [6],
            "ffmiddle": [7],
            "ffdistal": [8],
            "mfknuckle": [9],
            "mfproximal": [10],
            "mfmiddle": [11],
            "mfdistal": [12],
            "rfknuckle": [13],
            "rfproximal": [14],
            "rfmiddle": [15],
            "rfdistal": [16],
            "lfmetacarpal": [17],
            "lfknuckle": [18],
            "lfproximal": [19],
            "lfmiddle": [20],
            "lfdistal": [21],
            "thbase": [22],
            "thproximal": [23],
            "thhub": [24],
            "thmiddle": [25],
            "thdistal": [26]
        }

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, input_dim].
        """
        outputs = []
        for key, indices in self.sensor_map.items():
            inputs = x[:, indices].to(self.zero_token.device)
            if inputs.shape[-1] == 0:
                outputs.append(self.zero_token.expand(*inputs.shape[:-1], -1).unsqueeze(1))
            else:
                outputs.append(self.tokenizers[key](inputs).unsqueeze(1))

        return torch.cat(outputs, dim=1)



class BoTModel(nn.Module):
    def __init__(self, d_model, head, layer_num=16, mode='Mix'):
        super().__init__()
        self.tokenizer = BoT_Tokenizer(d_model)
        self.encoder = BoTEncoder(d_model, head, layer_num, mode)
        self.detokenizer = BoT_Detokenizer(d_model)

        # door-expert-v2 shortest_path_matrix converted to path_mask
        self.shortest_path_matrix = torch.Tensor(
            [[0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7],
             [1, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6],
             [2, 1, 0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
             [3, 2, 1, 0, 1, 2, 3, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6],
             [4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7],
             [5, 4, 3, 2, 1, 0, 1, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8],
             [6, 5, 4, 3, 2, 1, 0, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
             [3, 2, 1, 2, 3, 4, 5, 0, 1, 2, 3, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6],
             [4, 3, 2, 3, 4, 5, 6, 1, 0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7],
             [5, 4, 3, 4, 5, 6, 7, 2, 1, 0, 1, 4, 5, 6, 7, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8],
             [6, 5, 4, 5, 6, 7, 8, 3, 2, 1, 0, 5, 6, 7, 8, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
             [3, 2, 1, 2, 3, 4, 5, 2, 3, 4, 5, 0, 1, 2, 3, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6],
             [4, 3, 2, 3, 4, 5, 6, 3, 4, 5, 6, 1, 0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7],
             [5, 4, 3, 4, 5, 6, 7, 4, 5, 6, 7, 2, 1, 0, 1, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8],
             [6, 5, 4, 5, 6, 7, 8, 5, 6, 7, 8, 3, 2, 1, 0, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9],
             [3, 2, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 0, 1, 2, 3, 4, 2, 3, 4, 5, 6],
             [4, 3, 2, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 1, 0, 1, 2, 3, 3, 4, 5, 6, 7],
             [5, 4, 3, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 2, 1, 0, 1, 2, 4, 5, 6, 7, 8],
             [6, 5, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 3, 2, 1, 0, 1, 5, 6, 7, 8, 9],
             [7, 6, 5, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 4, 3, 2, 1, 0, 6, 7, 8, 9, 10],
             [3, 2, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4],
             [4, 3, 2, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 7, 1, 0, 1, 2, 3],
             [5, 4, 3, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 8, 2, 1, 0, 1, 2],
             [6, 5, 4, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 9, 3, 2, 1, 0, 1],
             [7, 6, 5, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 10, 4, 3, 2, 1, 0]]
        )
        self.path_mask = (self.shortest_path_matrix <= 1).float()
        #self.path_mask = self.path_mask.masked_fill(self.shortest_path_matrix >= 2, float('-inf'))

        self.path_mask = self.path_mask.to("cuda:0")

    def forward(self, x, mask=None):
        out = self.tokenizer(x)
        out, attn_map = self.encoder(out, mask=self.path_mask if mask is None else mask)
        out = self.detokenizer(out)
        return out, attn_map