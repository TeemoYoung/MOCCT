import torch
from torch import nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        #self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
   
        #attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)
        return output, attn
    
    pass

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_q, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_q, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        #self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):

        residual = q

        q = self.w_qs(q).view(-1, 3, self.n_head, self.d_q)
        k = self.w_ks(k).view(-1, 3, self.n_head, self.d_k)
        v = self.w_vs(v).view(-1, 3, self.n_head, self.d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v)

        q = q.transpose(1, 2).contiguous().view(-1, 3, 10)
        #q = self.dropout(self.fc(q))

        q += residual

        q = self.layer_norm(q)
        q = q.reshape(-1, 30)
        return q, attn
    
    pass


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        x += residual
        x = self.layer_norm(x)
        return x
    
    pass

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(90, 45)
        self.linear2 = nn.Linear(45, 3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    pass


class MOCCT(nn.Module):


    def __init__(self):
        super(MOCCT, self).__init__()
        self.multi_attention = MultiHeadAttention(1, 10, 10, 10, 10)
        self.PFF = PositionwiseFeedForward(30, 30)
        self.PFF1 = PositionwiseFeedForward(90, 90)
        self.cls = Classifier()

    def forward(self, q, k, v):

        q0 = q[:, :30].reshape(-1, 3, 10)
        k0 = k[:, :30].reshape(-1, 3, 10)
        v0 = v[:, :30].reshape(-1, 3, 10)

        q1 = q[:, 30:60].reshape(-1, 3, 10)
        k1 = k[:, 30:60].reshape(-1, 3, 10)
        v1 = v[:, 30:60].reshape(-1, 3, 10)

        q2 = q[:, 60:].reshape(-1, 3, 10)
        k2 = k[:, 60:].reshape(-1, 3, 10)
        v2 = v[:, 60:].reshape(-1, 3, 10)


        q01, attn01 = self.multi_attention(q0, k0, v0)
        q11, attn11 = self.multi_attention(q1, k1, v1)
        q21, attn21 = self.multi_attention(q2, k2, v2)
        q01 = self.PFF(q01)
        q11 = self.PFF(q11)
        q21 = self.PFF(q21)
        q01 = q01.reshape(-1, 3, 10)
        q11 = q11.reshape(-1, 3, 10)
        q21 = q21.reshape(-1, 3, 10)

        output0, attn0 = self.multi_attention(q01, q01, q01)
        output1, attn1 = self.multi_attention(q11, q11, q11)
        output2, attn2 = self.multi_attention(q21, q21, q21)

        output0 = self.PFF(output0)
        output1 = self.PFF(output1)
        output2 = self.PFF(output2)

 
        output = torch.cat((output0, output1, output2), dim=1)
        output = self.PFF1(output)

        result = self.cls(output)
        result = F.softmax(result, dim=1)
        return result
    
    pass

class BP(nn.Module):

    def __init__(self):
        super(BP, self).__init__()
        self.linear1 = nn.Linear(90, 45)
        self.linear2 = nn.Linear(45, 3)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.softmax(x, dim=1)
        return x
    pass