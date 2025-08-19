import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from mamba.models.mixer_seq_simple import MixerModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # query shape: [batch_size, query_len, d_model]
        # key shape: [batch_size, key_len, d_model]
        # value shape: [batch_size, value_len, d_model]

        batch_size = query.size(0)

        # project inputs to multi-heads
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, query_len, head_dim]
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, key_len, head_dim]
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2) # [batch_size, n_heads, value_len, head_dim]

        # compute scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # [batch_size, n_heads, query_len, key_len]

        # apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # apply softmax
        attention_weights = F.softmax(scores, dim=-1) # [batch_size, n_heads, query_len, key_len]
        out = torch.matmul(attention_weights, V) # [batch_size, n_heads, query_len, head_dim]

        # concatenate heads and apply final linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # [batch_size, query_len, d_model]
        out = self.fc_out(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]

        # self-attention
        attn_output = self.self_attn(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)

        # feed forward
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)

        return x
    
#=====================================================================#
#================================ MTPN ===============================#
#=====================================================================#

class Uncond_MTPN_EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, mamba_dim, n_heads, attn_layers, backbone, dropout_rate=0.1, device=None):
        super(Uncond_MTPN_EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(mamba_dim, d_model)

        self.device = device
        self.d_model = d_model
        self.backbone = backbone
        self.attn_layers = attn_layers

        self.global_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(attn_layers)]).to(device)
    
    def global_forward(self, x, mask=None):
        for layer in self.global_layers:
            x = layer(x, mask)

        return x

    def local_forward(self, x, model=None):
        # mamba
        seq_len = x.shape[1]
        batch_size = x.shape[0]

        if model == 'half':
            seq_len_temp = seq_len // 2
        elif model == 'quarter':
            seq_len_temp = seq_len // 4

        x = x[:, -seq_len_temp:, :]
        tensor_zero = torch.zeros(batch_size, seq_len-seq_len_temp, self.d_model, device=self.device)

        mamba_output = self.backbone(x)
        mamba_output = self.dropout1(mamba_output)
        mamba_output = self.linear(mamba_output)
        x = self.layer_norm1(x + mamba_output)
        x = torch.cat((tensor_zero, x), dim=1)

        return x

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, embedding_dim]
        # global_forward and local_forward
        if x.shape[1] < 2:
            output = self.global_forward(x, mask)
        elif 2 <= x.shape[1] < 4:
            global_out = self.global_forward(x, mask)
            local_out = self.local_forward(x, 'half')
            output = global_out + local_out
        else:
            global_out = self.global_forward(x, mask)
            local_out1 = self.local_forward(x, 'half')
            local_out2 = self.local_forward(x, 'quarter')
            output = global_out + local_out1 + local_out2

        x = self.layer_norm2(output + x)

        # feed forward
        ff_output = self.feed_forward(output)
        ff_output = self.dropout2(ff_output)
        x = self.layer_norm3(x + ff_output)

        return x
    
class Uncond_MTPN_Encoder(nn.Module):
    def __init__(
            self, 
            d_model, 
            d_ff,
            mamba_dim,
            backbone,
            n_heads=4,
            outter_layers=2,
            attn_layers=2,
            dropout_rate=0.1, 
            device=None,
        ):
        super(Uncond_MTPN_Encoder, self).__init__()
        self.layers = nn.ModuleList([Uncond_MTPN_EncoderLayer(d_model, d_ff, mamba_dim, n_heads, attn_layers, backbone, dropout_rate, device) for _ in range(outter_layers)]).to(device)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]

        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x)
    
class Uncond_MTPN_Policy(nn.Module):
    def __init__(
            self, 
            input_dim, 
            d_model,
            d_ff,
            mamba_dim=64,
            n_heads=4,
            mamba_layers=2, 
            outter_layers=2,
            attn_layers=2,
            dropout_rate=0.1, 
            ssm_cfg=None, 
            rms_norm: bool=True, 
            initializer_cfg=None, 
            fused_add_norm: bool=True, 
            residual_in_fp32: bool=True,
            device=None,
            dtype=None,
        ):
        super(Uncond_MTPN_Policy, self).__init__()

        self.backbone = MixerModel(
            d_model=mamba_dim,
            n_layer=mamba_layers,
            vocab_size=d_model,
            # dropout_val=self.dropout,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            device=device,
            dtype=dtype,
        ).to(device)

        self.device = device
        self.state_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.encoder = Uncond_MTPN_Encoder(d_model, d_ff, mamba_dim, self.backbone, n_heads, outter_layers, attn_layers, dropout_rate, device).to(device)

        self.state_out = nn.Sequential(nn.Linear(d_model, d_model),
                                       nn.ReLU(),
                                       nn.Linear(d_model, input_dim))
        
    def create_lower_triangle_mask(self, seq_len, batch_size):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))  
        mask = mask.bool().unsqueeze(0).unsqueeze(1)  # ->[1, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, -1, -1, -1)  # ->[batch_size, 1, seq_len, seq_len]
        return mask

    def forward(self, x, stage='train'):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        mask = self.create_lower_triangle_mask(seq_len, batch_size) if stage == 'train' else None           

        state_emb = self.state_embedding(x) 
        x = self.pos_encoding(state_emb)
        x = self.layer_norm(x)
        x = self.encoder(x, mask)

        next_state = self.state_out(x)

        return next_state

    def sample(self, x, horizon):
        for _ in range(horizon-1):
            n_x = self.forward(x, "plan")
            x = torch.cat([x, n_x[:, -1:]], dim=1)
        return x
    
#=====================================================================#
#============================= Transformer ===========================#
#=====================================================================#

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

class Uncond_Transformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, dropout_rate=0.1):
        super(Uncond_Transformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)

        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers, dropout_rate)
        self.state_out = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.ReLU(),
                                    nn.Linear(d_model, input_dim))
        self.embed_ln = nn.LayerNorm(d_model)


    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        # return shape: [batch_size, seq_len, input_dim]
        state_emb = self.embedding(x) # [batch_size, seq_len, d_model]

        x = self.pos_encoding(x)
        x = self.embed_ln(x)
        x = self.encoder(x, mask)

        state = self.state_out(x)

        return state
    
    def sample(self, x, horizon):
        for _ in range(horizon-1):
            n_x = self.forward(x)
            x = torch.cat([x, n_x[:, -1:]], dim=1)
        return x
    
#=====================================================================#
#=============================== Mamba ===============================#
#=====================================================================#

class Uncond_MambaPolicy(nn.Module):
    def __init__(
            self, 
            input_dim, 
            d_model, 
            mamba_dim,
            mamba_layers=4, 
            dropout_rate=0.1, 
            ssm_cfg=None, 
            rms_norm: bool=True, 
            initializer_cfg=None, 
            fused_add_norm: bool=True, 
            residual_in_fp32: bool=True,
            device=None,
            dtype=None,
        ):
        super(Uncond_MambaPolicy, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.mamba_dim = mamba_dim
        self.mamba_layers = mamba_layers
        self.dropout = dropout_rate

        self.state_embedding = nn.Linear(self.input_dim, self.d_model)

        self.pos_encoding = PositionalEncoding(self.d_model)
        self.embed_ln = nn.LayerNorm(self.d_model)
        self.state_out = nn.Sequential(nn.Linear(self.mamba_dim, int(self.mamba_dim/2)),
                                    nn.ReLU(),  # nn.PReLU()
                                    nn.Linear(int(self.mamba_dim/2), input_dim))
        
        factory_kwargs = {"device": device, "dtype": dtype}

        self.backbone = MixerModel(
            d_model=self.mamba_dim,
            n_layer=mamba_layers,
            vocab_size=self.d_model,
            # dropout_val=self.dropout,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        ).to(device)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # return shape: [batch_size, seq_len, input_dim]
        state_emb = self.state_embedding(x) # [batch_size, seq_len, d_model]
        x = self.pos_encoding(state_emb)
        x = self.embed_ln(x)
        x = self.backbone(x)

        next_state = self.state_out(x)

        return next_state
    
    def sample(self, x, horizon):
        for _ in range(horizon-1):
            n_x = self.forward(x)
            x = torch.cat([x, n_x[:, -1:]], dim=1)
        return x
    
