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
            #mask = mask.unsqueeze(1)
            score = score.masked_fill(mask == 0, float('-inf'))
        #identity = torch.eye(score.size(-1), device=score.device).unsqueeze(0).unsqueeze(0)
        #score = score + identity

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
        for i, layer in enumerate(self.layers):
            if self.mode == 'Mix' and i % 2 == 1:
                x, _ = layer(x, mask=None)
            else:
                x, _ = layer(x, mask)
        return x, _


class BoT_Detokenizer(nn.Module):
    def __init__(self, embodiment_num=6, d_model=512):
        super().__init__()
        self.regression_layer = nn.Linear(d_model, 1)
        self.classification_layers = nn.ModuleList([nn.Linear(d_model, 2) for _ in range(embodiment_num - 1)])

    def forward(self, x):
        outputs = []
        token_x = x[:, 0, :]#몸통은 전압
        regression_output = self.regression_layer(token_x)  # (batch_size, 1)
        outputs.append(regression_output)

        for i, layer in enumerate(self.classification_layers, start=1):
            token_x = x[:, i, :]#나머지는 행동이 위 아래 선택
            classification_output = layer(token_x)
            outputs.append(classification_output)

        # 모든 출력을 리스트로 반환
        return outputs

class BoT_Tokenizer(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.d_model = d_model
        self.body_embedding = nn.Linear(1, d_model)
        self.arm_embedding = nn.Embedding(4, d_model)

    def forward(self, body_sensor, arm_sensor):
        body_out = self.body_embedding(body_sensor)
        body_out = body_out.unsqueeze(1)
        arm_out = self.arm_embedding(arm_sensor)
        return torch.cat([body_out, arm_out], dim=1)

class BoTModel(nn.Module):
    def __init__(self, d_model, head, layer_num=6, embodiment_num=6, mode='Hard'):
        super().__init__()
        self.tokenizer = BoT_Tokenizer(d_model)
        self.encoder = BoTEncoder(d_model, head, layer_num, mode)
        self.detokenizer = BoT_Detokenizer(embodiment_num, d_model)
    def forward(self, body_sensor, arm_sensor, mask=None):
        out = self.tokenizer(body_sensor, arm_sensor)
        out, _ = self.encoder(out, mask)
        out = self.detokenizer(out)
        return out, _