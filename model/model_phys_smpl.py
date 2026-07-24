"""
PILA model (physics-informed low-rank augmentation).

Synced from upstream (github.com/yihshe/MAGIC) with local path adaptations.
DPM physics model support added for crop phenology inversion.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from base import BaseModel
from rtm_torch.rtm import RTM
from mogi.mogi import Mogi
from dpm.dpm import DPM
from dpm.dpm_stsc import DPM_STSC
from model import SCRIPT_DIR, PARENT_DIR
from utils import MLP, draw_normal

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

S2_FULL_BANDS = ['B01', 'B02_BLUE', 'B03_GREEN', 'B04_RED','B05_RE1', 
                 'B06_RE2', 'B07_RE3', 'B08_NIR1', 'B8A_NIR2', 'B09_WV', 'B10', 
                 'B11_SWI1', 'B12_SWI2']
SD = 500.0


class Encoders(nn.Module):
    def __init__(self, config:dict):
        super(Encoders, self).__init__()

        in_channels = config['arch']['args']['input_dim']
        no_phy = config['arch']['phys_vae']['no_phy']
        dim_z_aux = config['arch']['phys_vae']['dim_z_aux']
        dim_z_phy = config['arch']['phys_vae']['dim_z_phy']
        activation = config['arch']['phys_vae']['activation']
        num_units_feat = config['arch']['phys_vae']['num_units_feat']
        
        self.func_feat = FeatureExtractor(config)

        if dim_z_aux > 0:
            hidlayers_z_aux = config['arch']['phys_vae']['hidlayers_z_aux']
            self.func_z_aux_mean = MLP([num_units_feat,]+hidlayers_z_aux+[dim_z_aux,], activation)
            self.func_z_aux_lnvar = MLP([num_units_feat,]+hidlayers_z_aux+[dim_z_aux,], activation)

        if not no_phy:
            hidlayers_z_phy = config['arch']['phys_vae']['hidlayers_z_phy']
            self.func_z_phy_mean = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation)
            self.func_z_phy_lnvar = MLP([num_units_feat,]+hidlayers_z_phy+[dim_z_phy,], activation)


class Decoders(nn.Module):
    def __init__(self, config:dict):
        super(Decoders, self).__init__()

        in_channels = config['arch']['args']['input_dim']
        dim_z_aux = config['arch']['phys_vae']['dim_z_aux']
        dim_z_phy = config['arch']['phys_vae']['dim_z_phy']
        activation = config['arch']['phys_vae']['activation']
        no_phy = config['arch']['phys_vae']['no_phy']
        
        time_feat_dim = config['arch']['args'].get('time_feat_dim', 0)
        self.time_feat_dim = config['arch']['args'].get('time_feat_dim', 0)
        self.use_time_in_residual = config['arch']['args'].get('use_time_in_residual', False)

        if not no_phy:
            if dim_z_aux > 0:
                residual_rank = config['arch']['phys_vae'].get('residual_rank', dim_z_aux)
                
                coeff_input_dim = dim_z_aux + in_channels + (time_feat_dim if self.use_time_in_residual else 0)
                self.coeff = nn.Linear(coeff_input_dim, residual_rank, bias=True)
                self.s = nn.Parameter(torch.ones(residual_rank))
                self.B = nn.Parameter(torch.randn(in_channels, residual_rank))
                nn.init.orthogonal_(self.B)
                
                self.tau_init = config['arch']['phys_vae'].get('tau_init', 3.0)
                self.tau_final = config['arch']['phys_vae'].get('tau_final', 1.0)
                self.tau_warmup_epochs = config['arch']['phys_vae'].get('tau_warmup_epochs', 20)
                
                self.r_init = config['arch']['phys_vae'].get('r_init', 0.0)
                self.r_final = config['arch']['phys_vae'].get('r_final', 1.0)
                self.r_warmup_epochs = config['arch']['phys_vae'].get('r_warmup_epochs', 20)
                
                self.ortho_penalty_weight = config['arch']['phys_vae'].get('ortho_penalty_weight', 0.1)
            else:
                self.tau_init = 1.0
                self.tau_final = 1.0
                self.tau_warmup_epochs = 0
                self.r_init = 0.0
                self.r_final = 0.0
                self.r_warmup_epochs = 0
        else:
            if dim_z_aux > 0:
                self.func_aux_dec = MLP([dim_z_aux, 16, 32, 64, in_channels], activation)
            else:
                self.func_aux_dec = nn.Identity()

    def get_tau(self, epoch, epochs_pretrain):
        if epoch < epochs_pretrain:
            return self.tau_init
        elif epoch < epochs_pretrain + self.tau_warmup_epochs:
            progress = (epoch - epochs_pretrain) / self.tau_warmup_epochs
            return self.tau_init + progress * (self.tau_final - self.tau_init)
        return self.tau_final
    
    def get_r(self, epoch, epochs_pretrain):
        if epoch < epochs_pretrain:
            return self.r_init
        elif epoch < epochs_pretrain + self.r_warmup_epochs:
            progress = (epoch - epochs_pretrain) / self.r_warmup_epochs
            return self.r_init + progress * (self.r_final - self.r_init)
        return self.r_final
    
    def get_current_tau_r(self, epoch, epochs_pretrain):
        return {
            'tau': self.get_tau(epoch, epochs_pretrain),
            'r': self.get_r(epoch, epochs_pretrain),
            'epoch': epoch,
            'epochs_pretrain': epochs_pretrain
        }
    
    def set_tau_r_from_checkpoint(self, tau_r_dict):
        if tau_r_dict is not None:
            self._inference_tau = tau_r_dict.get('tau', self.tau_final)
            self._inference_r = tau_r_dict.get('r', self.r_final)
        else:
            self._inference_tau = self.tau_final
            self._inference_r = self.r_final
    
    def get_tau_for_inference(self):
        return getattr(self, '_inference_tau', self.tau_final)
    
    def get_r_for_inference(self):
        return getattr(self, '_inference_r', self.r_final)
    
    def compute_coefficient(self, z_aux, x_P_det, epoch, epochs_pretrain, use_inference_values=False, time_feats=None):
        if time_feats is not None and self.time_feat_dim > 0 and self.use_time_in_residual:
            time_feats_sliced = time_feats[..., :self.time_feat_dim]
            coeff_input = torch.cat([z_aux, x_P_det, time_feats_sliced], dim=1)
        else:
            coeff_input = torch.cat([z_aux, x_P_det], dim=1)
        c_raw = self.coeff(coeff_input)
        
        if use_inference_values:
            tau = self.get_tau_for_inference()
        else:
            tau = self.get_tau(epoch, epochs_pretrain)
        c = torch.tanh(c_raw / tau)
        return c
    
    def orthogonality_penalty(self):
        BtB = torch.matmul(self.B.T, self.B)
        I = torch.eye(self.B.shape[1], device=self.B.device)
        return torch.norm(BtB - I, p='fro') ** 2


class FeatureExtractor(nn.Module):
    def __init__(self, config:dict):
        super(FeatureExtractor, self).__init__()

        in_channels = config['arch']['args']['input_dim']
        hidlayers_feat = config['arch']['phys_vae']['hidlayers_feat']
        num_units_feat = config['arch']['phys_vae']['num_units_feat']
        activation = config['arch']['phys_vae']['activation']
        
        self.func_feat = MLP([in_channels,]+hidlayers_feat+[num_units_feat,], activation)

    def forward(self, x:torch.Tensor, t:torch.Tensor=None):
        # 方案 A：若输入为 (B, 7, 46)，展平为 (B, 322)
        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)
        return self.func_feat(x)


class Physics_RTM(nn.Module):
    def __init__(self, config:dict):
        super(Physics_RTM, self).__init__()
        self.model = RTM()
        self.z_phy_ranges = json.load(open(os.path.join(PARENT_DIR, config['arch']['args']['rtm_paras']), 'r'))
        self.bands_index = [i for i in range(
            len(S2_FULL_BANDS)) if S2_FULL_BANDS[i] not in ['B01', 'B10']]
        self.x_mean = torch.tensor(
            np.load(os.path.join(PARENT_DIR,config['arch']['args']['standardization']['x_mean']))
        ).float().unsqueeze(0).to(DEVICE)
        self.x_scale = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_scale']))
        ).float().unsqueeze(0).to(DEVICE)
    
    def rescale(self, z_phy:torch.Tensor):
        z_phy_rescaled = {}
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            z_phy_rescaled[para_name] = z_phy[:, i] * (
                self.z_phy_ranges[para_name]['max'] - self.z_phy_ranges[para_name]['min']
            ) + self.z_phy_ranges[para_name]['min']
        
        z_phy_rescaled['cd'] = torch.sqrt(
            (z_phy_rescaled['fc']*10000)/(torch.pi*SD))*2
        z_phy_rescaled['h'] = torch.exp(
            2.117 + 0.507*torch.log(z_phy_rescaled['cd'])) 
        
        return z_phy_rescaled
    
    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)
        if const is not None:
            z_phy_rescaled.update(const)
        output = self.model.run(**z_phy_rescaled)[:, self.bands_index]
        return (output - self.x_mean) / self.x_scale 


class Physics_Mogi(nn.Module):
    def __init__(self, config:dict):
        super(Physics_Mogi, self).__init__()

        self.z_phy_ranges = json.load(open(os.path.join(PARENT_DIR, config['arch']['args']['mogi_paras']), 'r'))
        self.station_info = json.load(open(os.path.join(PARENT_DIR, config['arch']['args']['station_info']), 'r'))
        
        x = torch.tensor([self.station_info[k]['xE']
                          for k in self.station_info.keys()])*1000
        y = torch.tensor([self.station_info[k]['yN']
                          for k in self.station_info.keys()])*1000
        self.model = Mogi(x,y)
        
        self.x_mean = torch.tensor(
            np.load(os.path.join(PARENT_DIR,config['arch']['args']['standardization']['x_mean']))
        ).float().unsqueeze(0).to(DEVICE)
        self.x_scale = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_scale']))
        ).float().unsqueeze(0).to(DEVICE)
    
    def rescale(self, z_phy:torch.Tensor):
        z_phy_rescaled = {}
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            minv = self.z_phy_ranges[para_name]['min']
            maxv = self.z_phy_ranges[para_name]['max']
            if len(z_phy.shape) == 3:
                z_phy_rescaled[para_name] = z_phy[:, :, i]*(maxv-minv)+minv
            else:
                z_phy_rescaled[para_name] = z_phy[:, i]*(maxv-minv)+minv

            if para_name in ['xcen', 'ycen', 'd']:
                z_phy_rescaled[para_name] = z_phy_rescaled[para_name]*1000

        z_phy_rescaled['dV'] = z_phy_rescaled['dV'] * \
            torch.pow(10, torch.tensor(5)) - torch.pow(10, torch.tensor(7))
        
        return z_phy_rescaled
    
    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)
        output = self.model.run(**z_phy_rescaled)
        return (output - self.x_mean) / self.x_scale 


class Physics_DPM(nn.Module):
    """Physics module for simplified DPM (shared summer crop + delta_eos)."""
    def __init__(self, config:dict):
        super(Physics_DPM, self).__init__()
        self.model = DPM()
        self.z_phy_ranges = json.load(open(os.path.join(PARENT_DIR, config['arch']['args']['dpm_paras']), 'r'))
        
        self.x_mean = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_mean']))
        ).float().unsqueeze(0).to(DEVICE)
        self.x_scale = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_scale']))
        ).float().unsqueeze(0).to(DEVICE)
    
    def rescale(self, z_phy:torch.Tensor):
        z_phy_rescaled = {}
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            minv = self.z_phy_ranges[para_name]['min']
            maxv = self.z_phy_ranges[para_name]['max']
            z_phy_rescaled[para_name] = z_phy[:, i] * (maxv - minv) + minv
        
        z_phy_rescaled['wheat_fraction'] = torch.clamp(
            z_phy_rescaled['wheat_fraction'], 0.05, 0.95)
        z_phy_rescaled['rice_fraction'] = torch.clamp(
            z_phy_rescaled['rice_fraction'], 0.01, 0.99)
        z_phy_rescaled['delta_eos'] = torch.clamp(
            z_phy_rescaled['delta_eos'], 5.0, 60.0)
        
        return z_phy_rescaled
    
    def constrained_z01(self, z_phy:torch.Tensor):
        """Return z in [0,1] space AFTER constraints are applied."""
        z_phy_rescaled = self.rescale(z_phy)
        z_out = torch.zeros_like(z_phy)
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            minv = self.z_phy_ranges[para_name]['min']
            maxv = self.z_phy_ranges[para_name]['max']
            z_out[:, i] = (z_phy_rescaled[para_name] - minv) / (maxv - minv + 1e-8)
        return z_out.clamp(1e-4, 1 - 1e-4)

    def forward(self, z_phy:torch.Tensor, const:dict=None):
        z_phy_rescaled = self.rescale(z_phy)
        output = self.model.run(**z_phy_rescaled)
        return (output - self.x_mean) / self.x_scale


class Physics_DPM_STSC(nn.Module):
    """Physics module for multi-band STSC forward model (output flattened to 322-D)."""

    def __init__(self, config: dict):
        super().__init__()
        em_path = config['arch']['args'].get('endmembers')
        if em_path is not None:
            em_path = os.path.join(PARENT_DIR, em_path)
        learnable = config['arch']['args'].get('learnable_endmembers', True)
        self.model = DPM_STSC(endmembers_path=em_path, learnable_endmembers=learnable)
        self.z_phy_ranges = json.load(
            open(os.path.join(PARENT_DIR, config['arch']['args']['dpm_paras']), 'r')
        )
        self.x_mean = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_mean']))
        ).float().unsqueeze(0).to(DEVICE)
        self.x_scale = torch.tensor(
            np.load(os.path.join(PARENT_DIR, config['arch']['args']['standardization']['x_scale']))
        ).float().unsqueeze(0).to(DEVICE)

    def rescale(self, z_phy: torch.Tensor):
        z_phy_rescaled = {}
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            minv = self.z_phy_ranges[para_name]['min']
            maxv = self.z_phy_ranges[para_name]['max']
            z_phy_rescaled[para_name] = z_phy[:, i] * (maxv - minv) + minv

        for k in ['wheat_frac', 'rice_frac', 'maize_frac', 'soy_frac']:
            if k in z_phy_rescaled:
                z_phy_rescaled[k] = torch.clamp(z_phy_rescaled[k], 0.0, 1.0)
        if 'flood_intensity' in z_phy_rescaled:
            z_phy_rescaled['flood_intensity'] = torch.clamp(
                z_phy_rescaled['flood_intensity'], 0.0, 1.0
            )
        return z_phy_rescaled

    def constrained_z01(self, z_phy: torch.Tensor):
        z_phy_rescaled = self.rescale(z_phy)
        z_out = torch.zeros_like(z_phy)
        for i, para_name in enumerate(self.z_phy_ranges.keys()):
            minv = self.z_phy_ranges[para_name]['min']
            maxv = self.z_phy_ranges[para_name]['max']
            z_out[:, i] = (z_phy_rescaled[para_name] - minv) / (maxv - minv + 1e-8)
        return z_out.clamp(1e-4, 1 - 1e-4)

    def forward(self, z_phy: torch.Tensor, const: dict = None):
        z_phy_rescaled = self.rescale(z_phy)
        output = self.model.run(**z_phy_rescaled)  # (B, 7, 46)
        flat = output.reshape(output.shape[0], -1)
        return (flat - self.x_mean) / self.x_scale


class PHYS_VAE_SMPL(nn.Module):
    def __init__(self, config:dict):
        super(PHYS_VAE_SMPL, self).__init__()

        self.no_phy = config['arch']['phys_vae']['no_phy']
        self.dim_z_aux = config['arch']['phys_vae']['dim_z_aux']
        self.dim_z_phy = config['arch']['phys_vae']['dim_z_phy']
        self.activation = config['arch']['phys_vae']['activation']
        self.in_channels = config['arch']['args']['input_dim']
        self.detach_x_P_for_bias = config['arch']['phys_vae'].get('detach_x_P_for_bias', True)

        self.use_ema_prior = config['trainer']['phys_vae'].get('use_ema_prior', False)
        self.ema_momentum = config['trainer']['phys_vae'].get('ema_momentum', 0.99)
        self.ema_min_var = 1e-3
        self.ema_max_var = 50.0

        self.enc = Encoders(config)
        self.dec = Decoders(config)
        self.physics_model = self.physics_init(config)
        
        if self.use_ema_prior and not self.no_phy:
            self.register_buffer('ema_mean', torch.zeros(self.dim_z_phy))
            self.register_buffer('ema_m2', torch.ones(self.dim_z_phy))
            self.register_buffer('ema_var', torch.ones(self.dim_z_phy))
        
        self.time_feats = None
        
    def physics_init(self, config:dict):
        physics_type = config['arch']['args']['physics']
        if physics_type == 'RTM':
            return Physics_RTM(config)
        elif physics_type == 'Mogi':
            return Physics_Mogi(config)
        elif physics_type == 'DPM':
            return Physics_DPM(config)
        elif physics_type == 'DPM_STSC':
            return Physics_DPM_STSC(config)
        else:
            raise ValueError(f"Unknown physics model type: {physics_type}")
    
    def generate_physonly(self, z_phy:torch.Tensor, const:dict=None):
        y = self.physics_model(z_phy, const=const)
        return y

    def update_ema_prior(self, u_phy_mean: torch.Tensor, u_phy_lnvar: torch.Tensor):
        if not self.use_ema_prior or self.no_phy:
            return
        
        with torch.no_grad():
            decay = self.ema_momentum
            mu_q = u_phy_mean.detach()
            var_q = torch.exp(u_phy_lnvar.detach())

            batch_mu = mu_q.mean(dim=0)
            batch_m2 = (var_q + mu_q**2).mean(dim=0)

            self.ema_mean.mul_(decay).add_(batch_mu, alpha=1 - decay)
            self.ema_m2.mul_(decay).add_(batch_m2, alpha=1 - decay)

            ema_var = (self.ema_m2 - self.ema_mean**2).clamp(self.ema_min_var, self.ema_max_var)
            self.ema_var.copy_(ema_var)

    def priors(self, n:int, device:torch.device):
        if self.use_ema_prior and not self.no_phy:
            ema_var = (self.ema_m2 - self.ema_mean**2).clamp(self.ema_min_var, self.ema_max_var)
            prior_u_phy_stat = {
                'mean': self.ema_mean.unsqueeze(0).expand(n, -1),
                'lnvar': ema_var.log().unsqueeze(0).expand(n, -1)
            }
        else:
            prior_u_phy_stat = {'mean': torch.zeros(n, self.dim_z_phy, device=device),
                                'lnvar': torch.zeros(n, self.dim_z_phy, device=device)}
        
        prior_z_aux_stat = {'mean': torch.zeros(n, max(0,self.dim_z_aux), device=device),
                            'lnvar': torch.zeros(n, max(0,self.dim_z_aux), device=device)}
        return prior_u_phy_stat, prior_z_aux_stat

    def encode(self, x:torch.Tensor, t:torch.Tensor=None):
        x_ = x
        n = x_.shape[0]
        device = x_.device

        self.time_feats = t

        feature = self.enc.func_feat(x_, t)

        if self.dim_z_aux > 0:
            z_aux_stat = {'mean': self.enc.func_z_aux_mean(feature),
                          'lnvar': self.enc.func_z_aux_lnvar(feature)}
        else:
            z_aux_stat = {'mean': torch.empty(n, 0, device=device),
                          'lnvar': torch.empty(n, 0, device=device)}

        if not self.no_phy:
            z_phy_stat = {'mean': self.enc.func_z_phy_mean(feature),
                          'lnvar': self.enc.func_z_phy_lnvar(feature)}
        else:
            z_phy_stat = {'mean': torch.empty(n, 0, device=device),
                          'lnvar': torch.empty(n, 0, device=device)}

        return z_phy_stat, z_aux_stat

    def draw(self, z_phy_stat:dict, z_aux_stat:dict, hard_z_phy:bool=False, hard_z_aux:bool=False):
        if not hard_z_phy:
            u_phy = draw_normal(z_phy_stat['mean'], z_phy_stat['lnvar'])
        else:
            u_phy = z_phy_stat['mean'].clone()
        
        if not hard_z_aux:
            z_aux = draw_normal(z_aux_stat['mean'], z_aux_stat['lnvar'])
        else:
            z_aux = z_aux_stat['mean'].clone()

        if not self.no_phy:
            z_phy = torch.sigmoid(u_phy)
        else:
            z_phy = torch.zeros(u_phy.shape[0], self.in_channels, device=u_phy.device)

        return z_phy, z_aux

    def decode(self, z_phy:torch.Tensor, z_aux:torch.Tensor, epoch:int=0, epochs_pretrain:int=20,
               full:bool=False, const:dict=None, use_inference_values:bool=False, detach_x_P_for_bias:bool=True):
        if not self.no_phy:
            y = self.physics_model(z_phy, const=const)
            x_P = y
            if self.dim_z_aux > 0:
                x_P_input = x_P.detach() if detach_x_P_for_bias else x_P
                c = self.dec.compute_coefficient(z_aux, x_P_input, epoch, epochs_pretrain, use_inference_values, self.time_feats)
                
                delta = torch.matmul(c * self.dec.s, self.dec.B.T)
                
                if use_inference_values:
                    r = self.dec.get_r_for_inference()
                else:
                    r = self.dec.get_r(epoch, epochs_pretrain)
                x_PB = x_P + r * delta
            else:
                x_PB = x_P.clone()
                delta = torch.zeros_like(x_P)
                c = torch.zeros(x_P.shape[0], 0, device=x_P.device)
        else:
            y = torch.zeros(z_phy.shape[0], self.in_channels, device=z_phy.device)
            if self.dim_z_aux > 0:
                x_PB = self.dec.func_aux_dec(z_aux) 
            else:
                x_PB = torch.zeros(z_phy.shape[0], self.in_channels, device=z_phy.device)
            x_P = x_PB.clone()
            delta = torch.zeros_like(x_PB)
            c = torch.zeros(x_P.shape[0], 0, device=x_P.device)

        if full:
            return x_PB, x_P, y, delta, c
        else:
            return x_PB

    def forward(self, x:torch.Tensor, t:torch.Tensor=None, reconstruct:bool=True, hard_z_phy:bool=False, hard_z_aux:bool=False,
                inference:bool=False, const:dict=None, epoch:int=0, epochs_pretrain:int=20):
        z_phy_stat, z_aux_stat = self.encode(x, t)

        if not reconstruct:
            return z_phy_stat, z_aux_stat
        
        if not inference:
            x_mean = self.decode(*self.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux), 
                                 epoch=epoch, epochs_pretrain=epochs_pretrain, full=False, const=const, use_inference_values=False, detach_x_P_for_bias=self.detach_x_P_for_bias)
            return z_phy_stat, z_aux_stat, x_mean
        else:
            z_phy, z_aux = self.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
            x_PB, x_P, _y, _delta, _c = self.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=epochs_pretrain, full=True, const=const, use_inference_values=True, detach_x_P_for_bias=self.detach_x_P_for_bias)
            return z_phy, z_aux, x_PB, x_P
