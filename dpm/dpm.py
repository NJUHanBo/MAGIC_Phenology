import numpy as np
import torch

class DPM():
    """
    Dynamic Phenology Model for wheat-rice/maize rotation mixed pixels.
    
    Core assumption: rice and maize in the same pixel share identical
    peak-season phenology (same climate & management), differing only
    in senescence timing — rice matures delta_eos days later than maize.
    
    Forward model (time-segmented mixing):
      DOY <= 170 (spring):  f_crop * wheat(t) + (1-f_crop) * background(t)
      DOY >  170 (summer):  f_crop * [f_rice * summer_rice(t) +
                                       (1-f_rice) * summer_maize(t)]
                             + (1-f_crop) * background(t)
    where:
      summer_maize(t) = phenology(summer params, eos=eos_base)
      summer_rice(t)  = phenology(summer params, eos=eos_base + delta_eos)
    
    Parameters: 15 total
      - wheat:   M, m, sos, mat, sen, eos (6)
      - summer:  M, m, sos, mat, sen, eos_base (6)
      - delta_eos: rice late-harvest extension in days (1)
      - wheat_fraction: crop area fraction (1)
      - rice_fraction: rice share within summer crop (1)
    """
    def __init__(self, time_points=None, **kwargs):
        super(DPM, self).__init__()
        if time_points is None:
            self.time_points = torch.arange(1, 362, 8)
        else:
            self.time_points = time_points

        self.background_params = {
            'base_evi': 0.3,
            'amplitude': 0.15,
            'peak_doy': 130
        }

    def phenology_model(self, t, M, m, sos, mat, sen, eos):
        """
        Double logistic phenology model: EVI(t) = (M-m)*(S1(t) - S2(t)) + m
        
        Parameters:
            t: time points tensor [time_points]
            M, m, sos, mat, sen, eos: phenology parameters tensor [batch_size]
            
        Returns:
            evi: EVI values tensor [batch_size, time_points]
        """
        t = t.unsqueeze(0)
        M = M.unsqueeze(1)
        m = m.unsqueeze(1)
        sos = sos.unsqueeze(1)
        mat = mat.unsqueeze(1) 
        sen = sen.unsqueeze(1)
        eos = eos.unsqueeze(1)
        
        exp_arg1 = 2 * (sos + mat - 2*t) / (mat - sos + 1e-6)
        exp_arg1 = torch.clamp(exp_arg1, -500, 500)
        S_sos_mat = 1 / (1 + torch.exp(exp_arg1))
        
        exp_arg2 = 2 * (sen + eos - 2*t) / (eos - sen + 1e-6)
        exp_arg2 = torch.clamp(exp_arg2, -500, 500)
        S_sen_eos = 1 / (1 + torch.exp(exp_arg2))
        
        evi = (M - m) * (S_sos_mat - S_sen_eos) + m
        evi = torch.clamp(evi, 0.0, 1.0)
        
        return evi

    def calculate_background_evi(self, t):
        """Background EVI (simplified cosine)."""
        seasonal_variation = self.background_params['amplitude'] * torch.cos(
            2 * torch.pi * (t - self.background_params['peak_doy']) / 365
        )
        background_evi = self.background_params['base_evi'] + seasonal_variation
        return torch.clamp(background_evi, 0.1, 0.6)

    def run(self, **paras):
        """
        Run forward model.
        
        Parameters:
            **paras: all 15 DPM parameters as tensors [batch_size]
                
        Returns:
            mixed_evi: [batch_size, n_time]
        """
        batch_size = paras['summer_M'].shape[0]
        device = paras['summer_M'].device
        time_points = self.time_points.to(device)
        
        wheat_evi = self.phenology_model(
            time_points,
            paras['wheat_M'], paras['wheat_m'],
            paras['wheat_sos'], paras['wheat_mat'],
            paras['wheat_sen'], paras['wheat_eos']
        )
        
        summer_maize = self.phenology_model(
            time_points,
            paras['summer_M'], paras['summer_m'],
            paras['summer_sos'], paras['summer_mat'],
            paras['summer_sen'], paras['summer_eos_base']
        )
        
        rice_eos = paras['summer_eos_base'] + paras['delta_eos']
        summer_rice = self.phenology_model(
            time_points,
            paras['summer_M'], paras['summer_m'],
            paras['summer_sos'], paras['summer_mat'],
            paras['summer_sen'], rice_eos
        )
        
        background_evi = self.calculate_background_evi(time_points)
        background_evi = background_evi.unsqueeze(0).expand(batch_size, -1)
        
        f_crop = paras['wheat_fraction'].unsqueeze(1)
        f_rice = paras['rice_fraction'].unsqueeze(1)
        
        summer_mix = f_rice * summer_rice + (1 - f_rice) * summer_maize
        
        mixed_evi = torch.zeros(batch_size, len(time_points), device=device)
        
        for i, doy in enumerate(time_points):
            if doy <= 170:
                mixed_evi[:, i] = (f_crop[:, 0] * wheat_evi[:, i] +
                                   (1 - f_crop[:, 0]) * background_evi[:, i])
            else:
                mixed_evi[:, i] = (f_crop[:, 0] * summer_mix[:, i] +
                                   (1 - f_crop[:, 0]) * background_evi[:, i])
        
        return mixed_evi
