import torch
import torch.nn as nn
import torch.optim as optim


from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config




class Generator(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers):
        super(Generator, self).__init__()
        
        config = Qwen2Config(2, hidden_size=hidden_size, intermediate_size=hidden_size*3, num_key_value_heads=8, num_attention_heads=8)
        self.down = nn.Linear(embedding_size, hidden_size) if embedding_size != hidden_size else nn.Identity()
        self.net = nn.ModuleList([Qwen2DecoderLayer(config, i) for i in range(num_layers)])
        self.up = nn.Linear(hidden_size, embedding_size) if embedding_size != hidden_size else nn.Identity()

    def forward(self, embedding, attention_mask):
        x = self.down(embedding)
        for layer in self.net:
            x = layer(x, attention_mask=attention_mask)[0]
        x = self.up(x)
        return x

    @classmethod
    def from_pretrained(cls, checkpoint_path, embedding_size, hidden_size, num_layers):
        model = cls(embedding_size, hidden_size, num_layers)
        model.load_state_dict(torch.load(checkpoint_path))
        return model


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
