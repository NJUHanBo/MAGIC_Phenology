"""Training script for PILA (Physics-Informed Low-Rank Augmentation)."""
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model import PHYS_VAE_SMPL
from parse_config import ConfigParser
from trainer import PhysVAETrainerSMPL
from utils import prepare_device
import wandb

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    data_loader = config.init_obj('data_loader', module_data)
    
    valid_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['data_dir_valid'],
        batch_size=64,
        shuffle=True,
        validation_split=0.0,
        num_workers=2,
        with_const=config['data_loader']['args'].get('with_const', False)
    )

    model = PHYS_VAE_SMPL(config)
    logger.info(model)

    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = None
    if 'lr_scheduler' in config.config:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = PhysVAETrainerSMPL(
        model, criterion, metrics, optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler
    )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PILA training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size'),
        CustomArgs(['--use_kl_term_z_phy'], type=str,
                   target='trainer;phys_vae;use_kl_term_z_phy'),
        CustomArgs(['--beta_max_z_phy'], type=float,
                   target='trainer;phys_vae;beta_max_z_phy'),
        CustomArgs(['--use_kl_term_z_aux'], type=str,
                   target='trainer;phys_vae;use_kl_term_z_aux'),
        CustomArgs(['--beta_max_z_aux'], type=float,
                   target='trainer;phys_vae;beta_max_z_aux'),
        CustomArgs(['--kl_warmup_epochs'], type=int,
                   target='trainer;phys_vae;kl_warmup_epochs'),
        CustomArgs(['--epochs_pretrain'], type=int,
                   target='trainer;phys_vae;epochs_pretrain'),
        CustomArgs(['--tau_warmup_epochs'], type=int,
                   target='arch;phys_vae;tau_warmup_epochs'),
        CustomArgs(['--tau_init'], type=float,
                   target='arch;phys_vae;tau_init'),
        CustomArgs(['--r_warmup_epochs'], type=int,
                   target='arch;phys_vae;r_warmup_epochs'),
        CustomArgs(['--r_init'], type=float,
                   target='arch;phys_vae;r_init'),
        CustomArgs(['--dim_z_aux'], type=int,
                   target='arch;phys_vae;dim_z_aux'),
        CustomArgs(['--residual_rank'], type=int,
                   target='arch;phys_vae;residual_rank'),
        CustomArgs(['--detach_x_P_for_bias'], type=str,
                   target='arch;phys_vae;detach_x_P_for_bias'),
        CustomArgs(['--edge_penalty_weight'], type=float,
                   target='trainer;phys_vae;edge_penalty_weight'),
        CustomArgs(['--use_ema_prior'], type=str,
                   target='trainer;phys_vae;use_ema_prior'),
        CustomArgs(['--loss'], type=str,
                   target='loss'),
    ]
    config = ConfigParser.from_args(args, options)
    
    wandb.init(
        project="PILA",
        name=f"{config['name']}",
        config=config,
        mode="online" if config['trainer']['wandb'] else "disabled",
    )

    main(config)

    wandb.finish()
