import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class HerbTransformerGenerator(nn.Module):
    def __init__(self, num_pathways, num_herbs, config):
        super().__init__()
        self.d_model = config['d_model']
        self.num_herbs = num_herbs
        
        self.disease_encoder = nn.Sequential(
            nn.Linear(num_pathways, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        self.herb_embedding = nn.Embedding(num_herbs + 1, self.d_model) 
        self.dosage_projector = nn.Linear(1, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=config['nhead'], batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config['num_layers'])
        
        self.herb_head = nn.Linear(self.d_model, num_herbs)
        
        self.dosage_head = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Softplus() 
        )
        with torch.no_grad():
            self.dosage_head[0].bias.fill_(2.0) 

    def forward(self, disease_vec, memory=None, tgt_seq=None):
        memory = self.disease_encoder(disease_vec).unsqueeze(1)
        if tgt_seq is None:
            batch_size = disease_vec.size(0)
            tgt_seq = torch.zeros(batch_size, 1, self.d_model).to(disease_vec.device)
        tgt_seq = self.pos_encoder(tgt_seq)
        output = self.transformer_decoder(tgt_seq, memory)
        last_step_output = output[:, -1, :]
        return self.herb_head(last_step_output), self.dosage_head(last_step_output), output