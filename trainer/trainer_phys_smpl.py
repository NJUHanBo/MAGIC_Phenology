"""
PILA trainer (physics-informed low-rank augmentation).

Synced from upstream (github.com/yihshe/MAGIC) with local adaptations.
Supports: low-rank residual monitoring, EMA prior, capacity control,
edge penalty, temporal smoothness, tau/r warmup scheduling.
"""
import numpy as np
import torch
import os
from torchvision.utils import make_grid
from base import BaseTrainer, PARENT_DIR
from utils import inf_loop, MetricTracker, kldiv_normal_normal
import wandb
from model.loss import mse_loss, mse_loss_per_channel


class PhysVAETrainerSMPL(BaseTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.device = device
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.no_phy = config['arch']['phys_vae']['no_phy']
        self.epochs_pretrain = config['trainer']['phys_vae'].get('epochs_pretrain', 0)
        self.dim_z_phy = config['arch']['phys_vae']['dim_z_phy']
        self.dim_z_aux = config['arch']['phys_vae']['dim_z_aux']

        self.beta_warmup = config['trainer']['phys_vae'].get('kl_warmup_epochs', 50)
        
        self.use_kl_term_z_phy = config['trainer']['phys_vae'].get('use_kl_term_z_phy', False)
        self.beta_max_z_phy = config['trainer']['phys_vae'].get('beta_max_z_phy', 1.0)
        self.use_kl_term_z_aux = config['trainer']['phys_vae'].get('use_kl_term_z_aux', False)
        self.beta_max_z_aux = config['trainer']['phys_vae'].get('beta_max_z_aux', 1.0)
        
        self.use_capacity_control = config['trainer']['phys_vae'].get('use_capacity_control', False)
        self.C_max = config['trainer']['phys_vae'].get('C_max', float(self.dim_z_phy))
        self.C_gamma = config['trainer']['phys_vae'].get('C_gamma', 10.0)
        self.beta_aux = config['trainer']['phys_vae'].get('beta_aux', 1.0)

        self.synthetic_data_loss_weight = config['trainer']['phys_vae'].get('balance_data_aug', 1.0)
        self.syn_loss_min_weight = config['trainer']['phys_vae'].get('syn_loss_min_weight', 0.1)
        self.syn_loss_decay_epochs = config['trainer']['phys_vae'].get('syn_loss_decay_epochs', 50)
        
        self.ortho_penalty_weight = config['trainer']['phys_vae'].get('ortho_penalty_weight', 0.1)
        self.coeff_penalty_weight = config['trainer']['phys_vae'].get('coeff_penalty_weight', 1e-4)
        self.delta_penalty_weight = config['trainer']['phys_vae'].get('delta_penalty_weight', 1e-4)
        
        self.edge_penalty_weight = config['trainer']['phys_vae'].get('edge_penalty_weight', 0.0)
        self.edge_penalty_power = config['trainer']['phys_vae'].get('edge_penalty_power', 1.0)
        
        self.temporal_smoothness_weight = config['trainer']['phys_vae'].get('temporal_smoothness_weight', 0.0)
        
        self.use_ema_prior = config['trainer']['phys_vae'].get('use_ema_prior', False)

        self.grad_clip_norm = config['trainer'].get('grad_clip_norm', 1.0)

        self.train_metrics = MetricTracker(
            'loss', 'rec_loss', 'kl_loss',
            'syn_data_loss',
            'residual_loss', 'residual_rel_diff',
            'ortho_penalty', 'coeff_penalty', 'delta_penalty',
            'edge_penalty', 'temporal_smoothness',
            'c_norm', 'delta_norm', 's_norm', 'basis_quality',
            'kl_u_phy', 'kl_z_aux',
            'beta_z_phy', 'beta_z_aux',
            'capacity_target', 'capacity_penalty',
            'ema_prior_mean_norm', 'ema_prior_std_mean',
            *[m.__name__ for m in metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'rec_loss', 'kl_loss', 'residual_loss', 'residual_rel_diff',
            *[m.__name__ for m in metric_ftns], writer=self.writer)

        self.data_key = config['trainer']['input_key']
        self.target_key = config['trainer']['output_key']
        self.input_const_keys = config['trainer'].get('input_const_keys', None)

        self.stablize_grad = config['trainer']['stablize_grad']
        self.stablize_count = 0

        self.log_u_stats = True
        self.initial_lr = self.optimizer.param_groups[0]['lr']

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        
        if epoch < self.epochs_pretrain:
            if self.optimizer.param_groups[0]['lr'] != self.initial_lr:
                self.optimizer.param_groups[0]['lr'] = self.initial_lr
        else:
            if epoch == self.epochs_pretrain:
                self.logger.info(f"Epoch {epoch}: Starting training phase, scheduler will control LR")
        
        sequence_len = None
        if not self.no_phy and epoch >= self.epochs_pretrain and self.use_kl_term_z_phy:
            training_epoch = epoch - self.epochs_pretrain
            beta_z_phy = self.beta_max_z_phy * self._linear_annealing_epoch(training_epoch, warmup_epochs=self.beta_warmup)
        else:
            beta_z_phy = 0.0
        
        if epoch >= self.epochs_pretrain and self.use_kl_term_z_aux:
            training_epoch = epoch - self.epochs_pretrain
            beta_z_aux = self.beta_max_z_aux * self._linear_annealing_epoch(training_epoch, warmup_epochs=self.beta_warmup)
        else:
            beta_z_aux = 0.0

        u_sum = None
        u2_sum = None
        u_count = 0

        for batch_idx, data_dict in enumerate(self.data_loader):
            data = data_dict[self.data_key].to(self.device)
            input_const = {k: data_dict[k].to(self.device) for k in self.input_const_keys} if self.input_const_keys else None
            
            time_feats = data_dict.get('time_feats', None)
            if time_feats is not None:
                time_feats = time_feats.to(self.device)
                if time_feats.dim() == 3:
                    time_feats = time_feats.view(-1, time_feats.size(-1))

            if data.dim() == 3:
                sequence_len = data.size(1)
                data = data.view(-1, data.size(-1))

            self.optimizer.zero_grad()

            z_phy_stat, z_aux_stat = self.model.encode(data, time_feats)

            if self.use_ema_prior and not self.no_phy and epoch > 1 + self.epochs_pretrain:
                self.model.update_ema_prior(z_phy_stat['mean'], z_phy_stat['lnvar'])

            hard_z_phy = not self.use_kl_term_z_phy
            hard_z_aux = not self.use_kl_term_z_aux
            z_phy, z_aux = self.model.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
            x_PB, x_P, y, delta, c = self.model.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=self.epochs_pretrain, full=True, const=input_const)

            rec_loss, kl_u_phy, kl_z_aux = self._vae_loss(data, z_phy_stat, z_aux_stat, x_PB)
            kl_loss = beta_z_phy * kl_u_phy + beta_z_aux * kl_z_aux

            residual_loss = torch.sum((x_PB - x_P).pow(2), dim=1).mean()
            residual_rel_diff = torch.mean(torch.abs(x_PB - x_P) / (torch.abs(x_P) + 1e-8)) * 100.0

            if not self.no_phy and epoch >= self.epochs_pretrain and self.config['arch']['phys_vae']['dim_z_aux'] > 0:
                ortho_penalty = self.model.dec.orthogonality_penalty()
                coeff_penalty = torch.sum(c.pow(2), dim=1).mean()
                delta_penalty = torch.sum(delta.pow(2), dim=1).mean()
            else:
                ortho_penalty = torch.tensor(0.0, device=data.device)
                coeff_penalty = torch.tensor(0.0, device=data.device)
                delta_penalty = torch.tensor(0.0, device=data.device)

            edge_penalty = self._edge_penalty(z_phy)
            temporal_smoothness = self._temporal_smoothness_loss(z_phy, sequence_len)

            # Compute synthetic data loss (parameter supervision via cycle consistency)
            syn_weight = 0.0
            synthetic_data_loss = torch.tensor(0.0, device=data.device)
            if not self.no_phy:
                synthetic_data_loss = self._synthetic_data_loss(data.shape[0])
                if epoch < self.epochs_pretrain:
                    syn_weight = self.synthetic_data_loss_weight
                else:
                    post_ep = epoch - self.epochs_pretrain
                    decay = max(self.syn_loss_min_weight,
                                self.synthetic_data_loss_weight * (0.5 ** (post_ep / max(1, self.syn_loss_decay_epochs))))
                    syn_weight = decay

            if epoch < self.epochs_pretrain:
                loss = syn_weight * synthetic_data_loss
                capacity_penalty = torch.tensor(0.0, device=data.device)
                C_t = 0.0
            else:
                if self.use_capacity_control and not self.no_phy and self.use_kl_term_z_phy:
                    training_epoch = epoch - self.epochs_pretrain
                    warm = max(1, self.beta_warmup)
                    C_t = min(self.C_max, self.C_max * training_epoch / warm)
                    capacity_penalty = self.C_gamma * (kl_u_phy - C_t)**2
                    aux_term = beta_z_aux * kl_z_aux
                    loss = (rec_loss + capacity_penalty + aux_term
                            + syn_weight * synthetic_data_loss
                            + self.ortho_penalty_weight * ortho_penalty
                            + self.coeff_penalty_weight * coeff_penalty
                            + self.delta_penalty_weight * delta_penalty
                            + self.edge_penalty_weight * edge_penalty
                            + self.temporal_smoothness_weight * temporal_smoothness)
                else:
                    loss = (rec_loss + kl_loss
                            + syn_weight * synthetic_data_loss
                            + self.ortho_penalty_weight * ortho_penalty
                            + self.coeff_penalty_weight * coeff_penalty
                            + self.delta_penalty_weight * delta_penalty
                            + self.edge_penalty_weight * edge_penalty
                            + self.temporal_smoothness_weight * temporal_smoothness)
                    C_t = 0.0
                    capacity_penalty = torch.tensor(0.0, device=data.device)

            loss.backward()

            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            if self.stablize_grad:
                self._grad_stablizer(epoch, batch_idx, loss.item())

            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('rec_loss', rec_loss.item())
            self.train_metrics.update('kl_loss', kl_loss.item())
            self.train_metrics.update('residual_loss', residual_loss.item())
            self.train_metrics.update('residual_rel_diff', residual_rel_diff.item())
            self.train_metrics.update('kl_u_phy', kl_u_phy.item())
            self.train_metrics.update('kl_z_aux', kl_z_aux.item())
            self.train_metrics.update('beta_z_phy', beta_z_phy)
            self.train_metrics.update('beta_z_aux', beta_z_aux)
            self.train_metrics.update('capacity_target', C_t)
            self.train_metrics.update('capacity_penalty', capacity_penalty.item())
            self.train_metrics.update('edge_penalty', edge_penalty.item())
            self.train_metrics.update('temporal_smoothness', temporal_smoothness.item())
            
            if self.use_ema_prior and not self.no_phy and hasattr(self.model, 'ema_mean') and epoch > 1 + self.epochs_pretrain:
                ema_mean_norm = torch.norm(self.model.ema_mean).item()
                ema_var = (self.model.ema_m2 - self.model.ema_mean**2).clamp(self.model.ema_min_var, self.model.ema_max_var)
                ema_std_mean = torch.mean(torch.sqrt(ema_var + 1e-8)).item()
                self.train_metrics.update('ema_prior_mean_norm', ema_mean_norm)
                self.train_metrics.update('ema_prior_std_mean', ema_std_mean)
            else:
                self.train_metrics.update('ema_prior_mean_norm', 0.0)
                self.train_metrics.update('ema_prior_std_mean', 1.0)
            
            self.train_metrics.update('syn_data_loss', syn_weight * synthetic_data_loss.item())
            if not self.no_phy and epoch < self.epochs_pretrain:
                pass
            else:
                self.train_metrics.update('ortho_penalty', ortho_penalty.item())
                self.train_metrics.update('coeff_penalty', coeff_penalty.item())
                self.train_metrics.update('delta_penalty', delta_penalty.item())
                
                c_norm = torch.norm(c, dim=1).mean().item() if c.numel() > 0 else 0.0
                delta_norm = torch.norm(delta, dim=1).mean().item()
                s_norm = torch.norm(self.model.dec.s).item() if hasattr(self.model.dec, 's') else 0.0
                basis_quality = torch.norm(torch.matmul(self.model.dec.B.T, self.model.dec.B) - torch.eye(self.model.dec.B.shape[1], device=self.model.dec.B.device), p='fro').item() if hasattr(self.model.dec, 'B') else 0.0
                
                self.train_metrics.update('c_norm', c_norm)
                self.train_metrics.update('delta_norm', delta_norm)
                self.train_metrics.update('s_norm', s_norm)
                self.train_metrics.update('basis_quality', basis_quality)

            if not self.no_phy and self.log_u_stats:
                u = z_phy_stat['mean'].detach()
                if u_sum is None:
                    u_sum = u.sum(dim=0)
                    u2_sum = (u**2).sum(dim=0)
                else:
                    u_sum += u.sum(dim=0)
                    u2_sum += (u**2).sum(dim=0)
                u_count += u.size(0)

            if batch_idx % self.config['trainer'].get('log_step', 10) == 0:
                log_str = (f"Train Ep {epoch} [{batch_idx}/{len(self.data_loader)}] "
                           f"Loss {loss.item():.6f} Rec {rec_loss.item():.6f} "
                           f"KL(phy={beta_z_phy:.3f},aux={beta_z_aux:.3f}) {kl_loss.item():.6f} "
                           f"Residual {residual_loss.item():.6f} "
                           f"rel_diff {residual_rel_diff.item():.2f}%")
                self.logger.info(log_str)

        log = self.train_metrics.result()

        current_lr = self.optimizer.param_groups[0]['lr']
        summary_str = (f"Epoch {epoch} Summary - "
                       f"Loss: {log['loss']:.6f}, Rec: {log['rec_loss']:.6f}, "
                       f"KL: {log['kl_loss']:.6f}, "
                       f"Residual: {log['residual_loss']:.6f}, "
                       f"rel_diff: {log['residual_rel_diff']:.2f}%, "
                       f"LR: {current_lr:.6f}")
        
        if not self.no_phy and self.dim_z_aux > 0:
            r_value = self.model.dec.get_r(epoch, self.epochs_pretrain)
            tau_value = self.model.dec.get_tau(epoch, self.epochs_pretrain)
            summary_str += f", r(t): {r_value:.3f}, tau: {tau_value:.3f}"
        
        self.logger.info(summary_str)

        wandb.log({f'train/{key}': value for key, value in log.items()})
        wandb.log({'train/lr': current_lr, 'train/epoch': epoch,
                   'train/beta_z_phy': beta_z_phy, 'train/beta_z_aux': beta_z_aux})

        if not self.no_phy and self.log_u_stats and u_count > 0:
            u_mean = (u_sum / u_count).cpu().numpy()
            u_var = (u2_sum / u_count).cpu().numpy() - u_mean**2
            u_std = np.sqrt(np.maximum(u_var, 1e-12))
            wandb.log({f'train/u_mean_dim{i}': float(u_mean[i]) for i in range(len(u_mean))})
            wandb.log({f'train/u_std_dim{i}': float(u_std[i]) for i in range(len(u_std))})

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            wandb.log({f'val/{key}': value for key, value in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        
        tau_r_values = None
        if not self.no_phy and self.dim_z_aux > 0:
            tau_r_values = self.model.dec.get_current_tau_r(epoch, self.epochs_pretrain)
        
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'tau_r_values': tau_r_values
        }
        
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, os.path.join(PARENT_DIR, filename))
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, os.path.join(PARENT_DIR, best_path))
            self.logger.info("Saving current best: model_best.pth ...")

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()

        total_rec_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, data_dict in enumerate(self.valid_data_loader):
                try:
                    data = data_dict[self.data_key].to(self.device)
                    input_const = {k: data_dict[k].to(self.device) for k in self.input_const_keys} if self.input_const_keys else None
                    
                    time_feats = data_dict.get('time_feats', None)
                    if time_feats is not None:
                        time_feats = time_feats.to(self.device)
                        if time_feats.dim() == 3:
                            time_feats = time_feats.view(-1, time_feats.size(-1))
                    
                    if data.dim() == 3:
                        data = data.view(-1, data.size(-1))

                    z_phy_stat, z_aux_stat = self.model.encode(data, time_feats)
                    hard_z_phy = not self.use_kl_term_z_phy
                    hard_z_aux = not self.use_kl_term_z_aux
                    z_phy, z_aux = self.model.draw(z_phy_stat, z_aux_stat, hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux)
                    x_PB, x_P, y, delta, c = self.model.decode(z_phy, z_aux, epoch=epoch, epochs_pretrain=self.epochs_pretrain, full=True, const=input_const)
                    
                    rec_loss, kl_u_phy, kl_z_aux = self._vae_loss(data, z_phy_stat, z_aux_stat, x_PB)
                    kl_loss = kl_u_phy + kl_z_aux
                    
                    residual_loss = torch.sum((x_PB - x_P).pow(2), dim=1).mean()
                    residual_rel_diff = torch.mean(torch.abs(x_PB - x_P) / (torch.abs(x_P) + 1e-8)) * 100.0

                    self.valid_metrics.update('rec_loss', rec_loss.item())
                    self.valid_metrics.update('kl_loss', kl_loss.item())
                    self.valid_metrics.update('residual_loss', residual_loss.item())
                    self.valid_metrics.update('residual_rel_diff', residual_rel_diff.item())
                    
                    total_rec_loss += rec_loss.item()
                    total_kl_loss += kl_loss.item()
                    num_batches += 1

                except Exception as e:
                    self.logger.warning(f"Error in validation batch {batch_idx}: {e}")
                    continue

        avg_rec_loss = total_rec_loss / max(num_batches, 1)
        avg_kl_loss = total_kl_loss / max(num_batches, 1)
        
        val_metrics = self.valid_metrics.result()
        self.logger.info(f"Validation Epoch: {epoch} Rec Loss: {avg_rec_loss:.6f} KL Loss: {avg_kl_loss:.6f} "
                         f"Residual: {val_metrics['residual_loss']:.6f} rel_diff: {val_metrics['residual_rel_diff']:.2f}%")
        return val_metrics

    def _vae_loss(self, data, z_phy_stat, z_aux_stat, x, pretrain=False):
        rec_loss = self.criterion(x, data)
        n = data.shape[0]
        prior_u_phy_stat, prior_z_aux_stat = self.model.priors(n, self.device)

        if not self.no_phy:
            KL_u_phy = kldiv_normal_normal(
                z_phy_stat['mean'], z_phy_stat['lnvar'],
                prior_u_phy_stat['mean'], prior_u_phy_stat['lnvar']
            ).mean()
        else:
            KL_u_phy = torch.zeros(n, device=self.device).mean()

        if pretrain or self.config['arch']['phys_vae']['dim_z_aux'] == 0:
            KL_z_aux = torch.zeros(n, device=self.device).mean()
        else:
            KL_z_aux = kldiv_normal_normal(
                z_aux_stat['mean'], z_aux_stat['lnvar'],
                prior_z_aux_stat['mean'], prior_z_aux_stat['lnvar']
            ).mean()

        return rec_loss, KL_u_phy, KL_z_aux

    def _synthetic_data_loss(self, batch_size):
        if not self.no_phy:
            self.model.eval()
            with torch.no_grad():
                z = torch.rand((batch_size, self.dim_z_phy), device=self.device).clamp(1e-4, 1-1e-4)
                synthetic_y = self.model.generate_physonly(z)
                z_constrained = self.model.physics_model.constrained_z01(z)
            self.model.train()
            synthetic_features = self.model.enc.func_feat(synthetic_y)
            inferred_u_phy = self.model.enc.func_z_phy_mean(synthetic_features)
            target_u = torch.log(z_constrained) - torch.log1p(-z_constrained)
            return torch.sum((inferred_u_phy - target_u).pow(2), dim=1).mean()
        else:
            return torch.zeros(1, device=self.device)

    def _edge_penalty(self, z_phy):
        if self.no_phy or z_phy is None:
            return torch.tensor(0.0, device=self.device)
        
        eps = 1e-6
        z_clamped = z_phy.clamp(eps, 1.0 - eps)
        edge_penalty = -torch.log(z_clamped) - torch.log(1.0 - z_clamped)
        edge_penalty = edge_penalty.pow(self.edge_penalty_power)
        return edge_penalty.mean()

    def _grad_stablizer(self, epoch, batch_idx, loss):
        para_grads = [v.grad.data for v in self.model.parameters(
        ) if v.grad is not None and torch.isnan(v.grad).any()]
        if len(para_grads) > 0:
            epsilon = 1e-7
            for v in para_grads:
                rand_values = torch.rand_like(v, dtype=torch.float)*epsilon
                mask = torch.isnan(v) | v.eq(0)
                v[mask] = rand_values[mask]
            self.stablize_count += 1
            self.logger.info(
                'epoch: {}, batch: {}, loss: {}, stablize count: {}'.format(
                    epoch, batch_idx, loss, self.stablize_count)
            )

    def _linear_annealing_epoch(self, epoch, warmup_epochs=30):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 1.0

    def _temporal_smoothness_loss(self, z_phy, sequence_len):
        if self.no_phy or z_phy is None or sequence_len is None:
            return torch.tensor(0.0, device=self.device)
        
        batch_size = z_phy.size(0) // sequence_len
        if batch_size * sequence_len != z_phy.size(0):
            return torch.tensor(0.0, device=self.device)
        
        z_phy_seq = z_phy.view(batch_size, sequence_len, -1)
        spatial_coords = z_phy_seq[:, :, :3]
        coord_variance = torch.var(spatial_coords, dim=1)
        return coord_variance.mean()
