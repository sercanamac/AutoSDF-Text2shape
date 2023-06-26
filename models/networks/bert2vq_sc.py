
import torch
import torch.nn as nn

from einops import rearrange
from models.networks.pvqvae_networks.modules import ResnetBlock as PVQVAEResnetBlock, AttnBlock, Normalize
from transformers import BertTokenizer, BertModel


class BERT2VQ(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bertmodel = BertModel.from_pretrained("bert-base-uncased")

        if opt.gpu_ids[0] != -1:
            self.device = f'cuda:{opt.gpu_ids[0]}'
        else:
            self.device + "cpu"
        ntoken=512
        nblocks = 4
        use_attn = False
        convt_layers = []
        
        self.dz = self.hz = self.wz = 8
        in_c = 512
        n_layers = opt.mlp_layers
        hidden_size = opt.mlp_hidden


        
        
        self.linear_in = nn.Linear(768, hidden_size)
        self.linear_to3d = nn.Linear(1024, self.hz * self.wz * self.dz)
        self.activation = nn.ReLU()
        self.linear3d_to_conv = torch.nn.Conv3d(1, in_c, 3, 1, 1)
        in_c +=1
        for i in range(nblocks):
            out_c = in_c
            convt_layers.append(PVQVAEResnetBlock(in_channels=in_c, out_channels=out_c, temb_channels=0, dropout=0.1,use_bn=True))
            if use_attn:
                convt_layers.append( AttnBlock(out_c) )
            in_c = out_c
        
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                  self.activation, nn.BatchNorm1d(hidden_size)) for i in range(n_layers)])
        


        
        self.convt_layers = nn.Sequential(*convt_layers)

        self.conv_out = torch.nn.Conv3d(in_c, ntoken, 3, 1, 1)
    
    def forward(self, x, z1):
        
        # Extract BERT Features
        tokenized = self.tokenizer(x,return_tensors='pt',padding=True).to(self.device)
        x = self.bertmodel(**tokenized).pooler_output
        z1 = z1.permute(0, 4, 1, 2, 3)
        # Map to 3D space
        x = self.linear_in(x)
        for l in self.mlp:
            x = l(x)
        
        x = self.linear_to3d(x).unsqueeze(1)
        x = rearrange(x, 'b c (d h w) -> b c d h w', d=8, h=8, w=8)


        # x = self.linear3d_to_conv(x)
        x = torch.cat([x, z1], axis=1)
        x = self.convt_layers(x)


        x = self.conv_out(x)

        return x