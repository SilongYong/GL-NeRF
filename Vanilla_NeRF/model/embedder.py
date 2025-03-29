import torch
import torch.nn as nn
from utils import spherical_harmonics

# Positional encoding (section 5.1)
class FourierEmbedder(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs, device=self.kwargs['device'])
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs, device=self.kwargs['device'])

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0, input_dims=1, device='cuda:0', embed_type='fourier'):
    if i == -1:
        return nn.Identity(), 3
    if embed_type == 'fourier':
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : input_dims,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
                    'device' : device
        }

        embedder_obj = FourierEmbedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        return embed, embedder_obj.out_dim
    else:
        raise NotImplementedError(f'no such type of embedder: {embed_type}')