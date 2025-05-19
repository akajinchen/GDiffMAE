from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch_sparse
from torch import nn

from Params import args
from Utils.Utils import calcRegLoss, pairPredict
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform_
zeroinit = nn.init.zeros_


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
        self.gcnLayers0 = nn.Sequential(*[GCNLayer() for i in range(args.gcn_layer0)])
        self.reset_parameters()

    def reset_parameters(self):
        for name, pama in self.named_parameters():
            if 'weight' in name:
                init(pama)
            elif 'bias' in name:
                zeroinit(pama)

    def getEgoEmbeds(self):
        return torch.cat([self.uEmbeds, self.iEmbeds], dim=0)
    
    def reparameter(self, mean, logvar):
        std = torch.exp(logvar / 2.)
        eps = torch.randn_like(std) * 0.1
        return mean + eps * std, eps


    def forward(self, adj, D_1adj):
        embeds_list = []
        embeds = self.getEgoEmbeds()
        embeds_list.append(embeds)
        for i, gcn in enumerate(self.gcnLayers0):
            embeds = gcn(adj, embeds)
            embeds_list.append(embeds)
        embeds = torch.stack(embeds_list, dim=0)
        embeds = torch.mean(embeds, dim=0)
        condition_embeds = gcn(D_1adj, embeds)

        return embeds[:args.user], embeds[args.user:], condition_embeds, self.uEmbeds, self.iEmbeds
        



    def getGCN(self):
        return self.gcnLayers

## lightgcn layer
class GCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, adj, embeds: torch.Tensor):
        if len(embeds.shape) == 2:
            return torch.sparse.mm(adj, embeds)
        
        if len(embeds.shape) == 3:
            k = embeds.shape[0]
            for i in range(k):
                h = embeds[i, :, :].squeeze(0)
                h = torch.sparse.mm(adj, h)
                if i == 0:
                    out = h.unsqueeze(0)
                else:
                    out = torch.cat((out, h.unsqueeze(0)), dim=0)
            return out
                    

class Diffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, time_step, Beta=False, beta=0.001,  history_num_per_term=10, ):
        super().__init__()
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.time_step = time_step
        self.norm_x = nn.LayerNorm(args.latdim, elementwise_affine=False)
        self.history_num_per_term = history_num_per_term
        self.Lt_history = torch.zeros(time_step, history_num_per_term, dtype=torch.float64).to(device)
        self.Lt_count = torch.zeros(time_step, dtype=int).to(device)
        

        if noise_scale != 0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(device)
            if Beta:
                self.betes[0] = beta
            self.calculate_for_diffusion()
        


    def get_betas(self):
        start = self.noise_scale * self.noise_min  
        end = self.noise_scale * self.noise_max    
        variance = np.linspace(start, end, self.time_step, dtype=np.float64)
        return variance
    
    def calculate_for_diffusion(self):
        alpha = torch.ones_like(self.betas, dtype=torch.float64) - self.betas
        self.alpha_bar_cumprod = torch.cumprod(alpha, axis=0)
        self.alpha_bar_cumprod_prev = torch.cat([torch.tensor([1.]).to(device), self.alpha_bar_cumprod[:-1]]).to(device)
        self.alpha_bar_cumprod_next = torch.cat([self.alpha_bar_cumprod[1:], torch.tensor([0.0]).to(device)]).to(device)
        assert self.alpha_bar_cumprod_prev.shape[0] == self.time_step  
        self.sqrt_alpha_bar_cumprod = torch.sqrt(self.alpha_bar_cumprod)
        self.sqrt_one_sub_alpha_bar_cumprod = torch.sqrt(1.0 - self.alpha_bar_cumprod)
        self.log_one_sub_alpha_bar_cumprod = torch.log(1.0 - self.alpha_bar_cumprod)
        self.sqrt_recip_alpha_bar_cumprod = torch.sqrt(1.0 / self.alpha_bar_cumprod)
        self.sqrt_recip_one_sub_alpha_bar_cumprod = torch.sqrt(1.0 / (self.alpha_bar_cumprod - 1.0))
        self.posterior_var = (
            self.betas * (1.0 - self.alpha_bar_cumprod_prev) / (1.0 - self.alpha_bar_cumprod)
        )
        self.posterior_log_var_clipped = torch.log(
            torch.cat([self.posterior_var[1].unsqueeze(0), self.posterior_var[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * (torch.sqrt(self.alpha_bar_cumprod_prev)) / (1.0 - self.alpha_bar_cumprod)
        )
        self.posterior_mean_coef2 = (
            torch.sqrt(alpha) * (1.0 - self.alpha_bar_cumprod_prev) / (1.0 - self.alpha_bar_cumprod)
        )


    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert x_start.shape == noise.shape
        return (self._extract_into_tensor(self.sqrt_alpha_bar_cumprod, t, x_start.shape) * x_start + 
                self._extract_into_tensor(self.sqrt_one_sub_alpha_bar_cumprod, t, noise.shape) * noise)

    def q_posterior_mean_var(self, x_start, xt, t):
        assert x_start.shape == xt.shape

        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_start.shape) * x_start + self._extract_into_tensor(self.posterior_mean_coef2, t, xt.shape) + xt
        )
        posterior_var = self._extract_into_tensor(self.posterior_var, t, x_start.shape)
        posterior_log_var_clip = self._extract_into_tensor(self.posterior_log_var_clipped, t, x_start.shape)

        assert (
            posterior_mean.shape[0] ==
            posterior_var.shape[0] ==
            posterior_log_var_clip.shape[0] == 
            x_start.shape[0]
        )
        return posterior_mean, posterior_var, posterior_log_var_clip


    def p_sample(self, model, x_start, condition_embeds, step, noise_d, sample_noise=args.sampleNoise):
        assert step <= self.time_step, "too much step"
        if step == 0:
            x_t = x_start
        else: 
            if noise_d is False:
                noise = torch.randn_like(x_start)
            else:
                miu, std = x_start.mean(dim=0), x_start.std(dim=0)
                noise = torch.randn_like(x_start)
                noise = noise * std + miu
                noise = self.norm_x(noise)
                noise = torch.abs(noise) * torch.sign(x_start)
            t = torch.tensor([step - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t, noise)
        indice = list(range(self.time_step))[::-1]
        
        if self.noise_scale == 0.:
            for i in indice:
                t = torch.tensor([i] * x_start.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t
        
        for i in indice:
            t = torch.tensor([i] * x_start.shape[0]).to(x_start.device)
            model_mean, model_log_var, model_var = self.p_mean_var(model, x_t, t, condition_embeds, True)
            uncond_model_mean, uncond_model_log_var, uncond_model_var = self.p_mean_var(model, x_t, t, None, False)

            if sample_noise:
                noise = torch.randn_like(model_mean) * 0.0001
                nozero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                uncond_model_sample = uncond_model_mean + nozero_mask * noise * torch.exp(0.5 * uncond_model_log_var)
                model_sample = model_mean + nozero_mask * noise * torch.exp(0.5 * model_log_var)
                x_t = uncond_model_sample + args.scale * (model_sample - uncond_model_sample)
            else:
                x_t = uncond_model_mean + args.scale * (model_mean - uncond_model_mean)
        return x_t

    def p_mean_var(self, model, x, t, condition_embeds, condition):
        model_output = model(x, t, condition_embeds, condition)

        model_var = self.posterior_var
        model_log_var = self.posterior_log_var_clipped

        model_var = self._extract_into_tensor(model_var, t, x.shape)
        model_log_var = self._extract_into_tensor(model_log_var, t, x.shape)

        model_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, model_output.shape) * model_output 
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x
        )

        return model_mean, model_log_var, model_var
    


    def training_loss(self, model, x_start, condition_embeds, noise_d, batch):
        batch_size = x_start.shape[0]
        ts = torch.randint(0, self.time_step, (batch_size,), device=device).long()
        if noise_d is False:
            noise = torch.randn_like(x_start)
        else:
            miu, std = x_start.mean(dim=0), x_start.std(dim=0)
            noise = torch.randn_like(x_start)
            noise = noise * std + miu
            noise = self.norm_x(noise)
            noise = torch.abs(noise) * torch.sign(x_start)
        
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
            model_out = model(x_t[batch], ts[batch], None, False)
            model_condition_out = model(x_t[batch], ts[batch], condition_embeds[batch], True)
        else:
            x_t = x_start
        mse = self.mean_flat((x_start[batch].detach() - model_out) ** 2)
        mse_condition = self.mean_flat((x_start[batch].detach() - model_condition_out) ** 2)
        weight = self.SNR(ts-1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)
        diff_loss = weight[batch] * (mse + mse_condition) * 0.5
        return diff_loss, model_out, model_condition_out


    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance': 
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')
            
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1- uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt
        
        elif method == 'uniform': 
            t = torch.randint(0, self.time_step, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()

            return t, pt
            
        else:
            raise ValueError


    def mean_flat(self, tensor):
        return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))

    def SNR(self, t):
        return self.alpha_bar_cumprod[t] / (1 - self.alpha_bar_cumprod[t])


    

class Denoise_NN(nn.Module):

    def __init__(self, in_dims, out_dims, emb_size, time_type='cat', norm=args.norm, act_func=args.actFunc, dropout=args.dropout, residual=args.residual):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], 'The input and output dimensions must be matched!'

        self.time_emb_dim = emb_size
        self.time_type = time_type
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if time_type == 'cat':
            in_dims_tmp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
            in_dims_tmp_condition = [2 * self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        out_dims_tmp = self.out_dims

        self.in_modeles = []
        self.condi_in_modeles = []
        self.user_in_modeles = []
        self.item_in_modeles = []
        for in_din, out_dim, in_dim_condi, out_dim_condi in zip(in_dims_tmp[:-1], in_dims_tmp[1:], in_dims_tmp_condition[:-1], in_dims_tmp_condition[1:]):
            self.in_modeles.append(nn.Linear(in_din, out_dim))
            self.condi_in_modeles.append(nn.Linear(in_dim_condi, out_dim_condi))
            self.user_in_modeles.append(nn.Linear(in_din, out_dim))
            self.item_in_modeles.append(nn.Linear(in_din, out_dim))
            if act_func == 'tanh':
                self.in_modeles.append(nn.Tanh())
                self.condi_in_modeles.append(nn.Tanh())
                self.user_in_modeles.append(nn.Tanh())
                self.item_in_modeles.append(nn.Tanh())
            if act_func == 'relu':
                self.in_modeles.append(nn.ReLU())
                self.condi_in_modeles.append(nn.ReLU())
            if act_func == 'sigmoid':   
                self.in_modeles.append(nn.Sigmoid())
                self.condi_in_modeles.append(nn.Sigmoid())
            if act_func == 'leakyrelu':
                self.in_modeles.append(nn.LeakyReLU())
                self.condi_in_modeles.append(nn.LeakyReLU())
            if act_func == 'Mish':
                self.in_modeles.append(nn.Mish())
                self.condi_in_modeles.append(nn.Mish())
        self.in_layer = nn.Sequential(*self.in_modeles)
        self.condi_in_layer = nn.Sequential(*self.condi_in_modeles)
        self.user_in_layer = nn.Sequential(*self.user_in_modeles)
        self.item_in_layer = nn.Sequential(*self.item_in_modeles)

        self.out_modeles = []
        self.condi_out_modeles = []
        self.user_out_modeles = []
        self.item_out_modeles = []
        for in_din, out_dim in zip(out_dims_tmp[:-1], out_dims_tmp[1:]):
            self.out_modeles.append(nn.Linear(in_din, out_dim))
            self.condi_out_modeles.append(nn.Linear(in_din, out_dim))
            self.user_out_modeles.append(nn.Linear(in_din, out_dim))
            self.item_out_modeles.append(nn.Linear(in_din, out_dim))
            if act_func == 'tanh':
                self.out_modeles.append(nn.Tanh())
                self.condi_out_modeles.append(nn.Tanh())
                self.user_out_modeles.append(nn.Tanh())
                self.item_out_modeles.append(nn.Tanh())
            if act_func == 'relu':
                self.out_modeles.append(nn.ReLU())
                self.condi_out_modeles.append(nn.ReLU())
            if act_func == 'sigmoid':   
                self.out_modeles.append(nn.Sigmoid())
                self.condi_out_modeles.append(nn.Sigmoid())
            if act_func == 'leakyrelu':
                self.out_modeles.append(nn.LeakyReLU())
                self.condi_out_modeles.append(nn.LeakyReLU())
            if act_func == 'Mish':
                self.out_modeles.append(nn.Mish())
                self.condi_out_modeles.append(nn.Mish())
        self.out_modeles.pop()
        self.condi_out_modeles.pop()
        self.user_out_modeles.pop()
        self.item_out_modeles.pop()
        self.out_layer = nn.Sequential(*self.out_modeles)
        self.condi_out_layer = nn.Sequential(*self.out_modeles)
        self.user_out_layer = nn.Sequential(*self.user_out_modeles)
        self.item_out_layer = nn.Sequential(*self.item_out_modeles)


        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(args.dropout1)

        self.reset_parameter()

    def reset_parameter(self):
        for name, parm in self.named_parameters():
            if 'weight' in name:
                init(parm)
            elif 'bias' in name:
                zeroinit(parm)

    def time_embedding(self, time, emb_dim, max_period=10000):
        """
        Creat time embeddings
        """
        half = emb_dim // 2
        freq = torch.exp(- math.log(max_period) * torch.arange(start=0, end=half,dtype=torch.float32) / half).to(time.device)
        args = time[:, None] * freq[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


    def forward(self, x, time, condition_embeds, condition=False):
        t_emb = self.time_embedding(time, self.time_emb_dim).to(x.device)
        t_emb = self.emb_layer(t_emb)
        _x = x
        if self.norm:
            x = F.normalize(x, dim=-1)
        x = self.dropout(x)
        
        if condition is True:
            x = torch.cat([x, condition_embeds], dim=-1)
            x = torch.cat([x, t_emb], dim=-1)
            x = self.condi_in_layer(x)
            x = self.dropout1(x)
            x = self.condi_out_layer(x)
        else:
            x = torch.cat([x, t_emb], dim=-1)
            x = self.in_layer(x)
            x = self.dropout1(x)
            x = self.out_layer(x)
        if args.residual:
            return x + _x
        return x
       

        