import omegaconf
import torch.nn as nn
from variation_net.modules import Attention
from prompt_tts.variation_net.nets import Variation_Net, Model
from conditioner import BERTConditioner

class StyleModule(nn.Module):
    def __init__(self, name: str, output_dim: int, 
                 prompt_seq_len: int, mert_seq_len: int, 
                 dim_head: int, heads: int, depth: int,  
                 device: str, timesteps: int=1000, causal: bool=False, 
                 use_ddim: bool=True, noise_schedule='sigmoid',
                 objective: str='v', schedule_kwargs:dict = dict(), time_difference=0., 
                 scale=1.):
        
        super().__init__()
        self.device = device
        
        self.bert = BERTConditioner(name=name, output_dim=output_dim, finetune=True,
                                    device=device, normalize_text=True)
        self.prompt_q = nn.Parameter(prompt_seq_len, output_dim)
        self.mert_q = nn.Parameter(mert_seq_len, output_dim)
        
        self.prompt_attn = Attention(dim=output_dim, dim_context=output_dim,
                                  dim_head=dim_head, heads=heads)  
        self.mert_attn = Attention(dim=output_dim, dim_context=output_dim,
                                  dim_head=dim_head, heads=heads)
        
        self.variation_net = Variation_Net(model=Model(dim=output_dim, depth=depth, causal=causal,
                                                       dim_head=dim_head, heads=heads, use_flash=True,),
                                           dim=output_dim, timesteps=timesteps, use_ddim=use_ddim, noise_schedule=noise_schedule,
                                           objective=objective, schedule_kwargs=schedule_kwargs,
                                           time_difference=time_difference, scale=scale)
    
    def forward(self, prompts, mert_feat):
        '''
        prompts: str 
        mert_feat: [b, T, 1024]
        '''
        
        prompt_tokens = self.bert.tokenize(prompts)
        prompt_enc, mask = self.bert(prompt_tokens) # [b, t, 1024]
        prompt_rep = self.prompt_attn(x=self.prompt_q.expand(prompt_enc.size(0), -1, -1),
                                   context=prompt_enc)
        mert_rep = self.mert_attn(x=self.mert_q.expand(mert_feat.size(0), -1, -1),
                                context=mert_feat)
        variation_loss = self.variation_net(mert_rep, prompt_rep)
        
        return variation_loss, prompt_rep, mert_rep
    
    def infer(self, prompt): ...