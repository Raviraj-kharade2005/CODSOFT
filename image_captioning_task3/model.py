import torch
import torch.nn as nn
from torchvision import models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return self.fc(features)

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads, hidden_size, num_layers):
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)
        self.pos_encoder = nn.Parameter(torch.rand(1, 100, embed_size))

    def forward(self, tgt, memory):
        tgt_emb = self.embed(tgt) + self.pos_encoder[:, :tgt.size(1), :]
        output = self.transformer(tgt_emb.transpose(0, 1), memory.unsqueeze(0))
        return self.fc(output.transpose(0, 1))
