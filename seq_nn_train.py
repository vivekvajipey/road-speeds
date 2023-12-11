import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import os, argparse, yaml

from dataset import RoadMeanSTDDataset, RoadMeanSTDSeqDataset
from models import ResNetConcat
from models.resnet_concat_rnn import ResNetConcatRNN
from util import set_seed, get_resnet, get_channel_dim, get_tabular_dim
from transforms import TABULAR_TRANSFORMS, TRAIN_TRANSFORMS, VAL_TRANSFORMS

import wandb

def denormalize(tensor, mean, std):
    return tensor * std + mean

def construct_run_name(args):
    name_pretrained = f'-pretrain-{args.pretrain}' if args.pretrain is not None else ''
    name_dataset = 'years' if 'year' in args.csv_path else 'month'
    
    name_arch = 'seq'
    name_seq = f"-{args.rnn_type}{args.hidden_dim}_layers{args.rnn_layers}"
    name_bands = ''.join([img_col.split('_')[0] for img_col in args.image_cols])
    name_hidden_dim = f"-hd{args.hidden_dim}" if args.hidden_dim else ""
    name_norm = 'bn'
    name_decay = f'dec{args.weight_decay}'
    name_dropout = f"d{args.dropout_rate}"
    name_std = '-nostd' if args.no_std else ''
    
    run_name = f"{args.model}-{name_arch}{name_seq}{name_pretrained}-{name_dataset}-mf-{name_bands}-{name_norm}{name_hidden_dim}-E{args.n_epochs}-B{args.batch_size}-Lr{args.lr}-{name_decay}-{name_dropout}{name_std}"
    return run_name

def main(args):
    set_seed(0)
    run_name = construct_run_name(args)
    if args.use_wandb:
        wandb.init(project="cs325b-road-quality", name=run_name, config=args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    with open(args.norm_params_file, 'r') as file:
        norm_params_yaml_data = yaml.safe_load(file)
    norm_params = norm_params_yaml_data[args.csv_path]

    tabular_cols = yaml.safe_load(open(args.tabular_cols_file))

    model = ResNetConcatRNN(
        resnet=get_resnet(args.model, args.pretrain),
        channel_dim=get_channel_dim(args.image_cols),
        tabular_dim=get_tabular_dim(tabular_cols),
        out_dim=2,
        hidden_dim=args.hidden_dim,
        rnn_layers=args.rnn_layers,
        rnn_type=args.rnn_type,
        pretrain_type=args.pretrain,
        dropout_rate=args.dropout_rate,
        freeze_enc=args.freeze_enc,
    )
    train_data = RoadMeanSTDSeqDataset(
        csv_path=args.csv_path,
        image_cols=args.image_cols,
        image_transform=TRAIN_TRANSFORMS,
        tabular_cols=tabular_cols,
        tabular_transform=TABULAR_TRANSFORMS,
        split='train',
    )
    val_data = RoadMeanSTDSeqDataset(
        csv_path=args.csv_path,
        image_cols=args.image_cols,
        image_transform=VAL_TRANSFORMS,
        tabular_cols=tabular_cols,
        tabular_transform=TABULAR_TRANSFORMS,
        split='val',
    )
    model = model.to(device)
    if args.use_wandb:
        wandb.watch(model)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion_mean = torch.nn.MSELoss(reduction='none') # no reduction to help calculate rmse
    criterion_std = torch.nn.MSELoss(reduction='none')
    if args.freeze_enc:
        params = model.get_head_parameters()
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    train_losses_mean = []
    train_losses_std = []
    val_losses_mean = []
    val_losses_std = []

    rmse_mean_train_list = []
    rmse_std_train_list = []
    mae_mean_train_list = []
    mae_std_train_list = []
    r2_mean_train_list = []
    r2_std_train_list = []

    rmse_mean_val_list = []
    rmse_std_val_list = []
    mae_mean_val_list = []
    mae_std_val_list = []
    r2_mean_val_list = []
    r2_std_val_list = []

    n_batches_per_train_epoch = len(train_data) // args.batch_size + 1
    n_batches_per_val_epoch = len(val_data) // args.batch_size + 1

    best_mean_rmse = float('inf')

    for epoch in range(args.n_epochs):
        model.train()

        epoch_mean_preds = []
        epoch_std_preds = []
        epoch_mean_labels = []
        epoch_std_labels = []

        for batch_idx, (img_batch, tab_batch, label_batch) in enumerate(tqdm(train_loader)):
            img_batch = img_batch.to(device)
            tab_batch = tab_batch.to(device)
            label_batch = label_batch.to(device)
            
            optimizer.zero_grad()
            outs = model(img_batch, tab_batch)

            outs_mean, outs_std = outs[:, :, 0], outs[:, :, 1]
            label_mean, label_std = label_batch[:, :, 0], label_batch[:, :, 1]

            loss_mean = (criterion_mean(outs_mean, label_mean).mean(dim=1)).mean()
            loss_std = (criterion_std(outs_std, label_std).mean(dim=1)).mean()
            if args.no_std:
                loss = loss_mean
            else:
                loss = loss_mean + loss_std

            loss.backward()
            optimizer.step()

            train_losses_mean.append(loss_mean.detach().cpu().item())
            train_losses_std.append(loss_std.detach().cpu().item())
            tqdm.write(f'Train Epoch {epoch} | MSE Mean {train_losses_mean[-1]:.4g} | MSE STD {train_losses_std[-1]:.4g}')

            if args.use_wandb:
                wandb.log({
                    "train_losses_mean": train_losses_mean[-1], 
                    "train_losses_std": train_losses_std[-1],
                    "epoch": epoch + batch_idx / n_batches_per_train_epoch,
                })

            # Yes Label Normalization
            label_mean_denorm = denormalize(label_mean, norm_params['speed_mean']['mean'], norm_params['speed_mean']['std']).detach().cpu().tolist()
            label_std_denorm = denormalize(label_std, norm_params['speed_std']['mean'], norm_params['speed_std']['std']).detach().cpu().tolist()
            outs_mean_denorm = denormalize(outs_mean, norm_params['speed_mean']['mean'], norm_params['speed_mean']['std']).detach().cpu().tolist()
            outs_std_denorm = denormalize(outs_std, norm_params['speed_std']['mean'], norm_params['speed_std']['std']).detach().cpu().tolist()

            epoch_mean_labels.extend(label_mean_denorm)
            epoch_std_labels.extend(label_std_denorm)
            epoch_mean_preds.extend(outs_mean_denorm)
            epoch_std_preds.extend(outs_std_denorm)

        epoch_mean_preds = np.array(epoch_mean_preds)
        epoch_std_preds = np.array(epoch_std_preds)
        epoch_mean_labels = np.array(epoch_mean_labels)
        epoch_std_labels = np.array(epoch_std_labels)

        epoch_mean_preds = epoch_mean_preds.flatten()
        epoch_std_preds = epoch_std_preds.flatten()
        epoch_mean_labels = epoch_mean_labels.flatten()
        epoch_std_labels = epoch_std_labels.flatten()

        epoch_mean_rmse = np.sqrt(np.mean(((epoch_mean_preds - epoch_mean_labels) ** 2)))
        epoch_std_rmse = np.sqrt(np.mean(((epoch_std_preds - epoch_std_labels) ** 2)))
        epoch_mean_mae = mean_absolute_error(epoch_mean_labels, epoch_mean_preds)
        epoch_std_mae = mean_absolute_error(epoch_std_labels, epoch_std_preds)
        epoch_mean_r2 = r2_score(epoch_mean_labels, epoch_mean_preds)
        epoch_std_r2 = r2_score(epoch_std_labels, epoch_std_preds)

        rmse_mean_train_list.append(epoch_mean_rmse)
        rmse_std_train_list.append(epoch_std_rmse)
        mae_mean_train_list.append(epoch_mean_mae)
        mae_std_train_list.append(epoch_std_mae)
        r2_mean_train_list.append(epoch_mean_r2)
        r2_std_train_list.append(epoch_std_r2)

        print((
            f'\nEpoch Train Mean RMSE {rmse_mean_train_list[-1]:.4g}, '
            f'MAE {mae_mean_train_list[-1]:.4g}, '
            f'R2 {r2_mean_train_list[-1]:.4g}'
        ))
        print((
            f'Epoch Train STD RMSE {rmse_std_train_list[-1]:.4g}, '
            f'MAE {mae_std_train_list[-1]:.4g}, '
            f'R2 {r2_std_train_list[-1]:.4g}\n'
        ))

        if args.use_wandb:
            wandb.log({
                "train_mean_rmse": rmse_mean_train_list[-1],
                "train_std_rmse": rmse_std_train_list[-1],
                "train_mean_mae": mae_mean_train_list[-1],
                "train_std_mae": mae_std_train_list[-1],
                "train_mean_r2": r2_mean_train_list[-1],
                "train_std_r2": r2_std_train_list[-1],
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

        model.eval()

        epoch_mean_preds = []
        epoch_std_preds = []
        epoch_mean_labels = []
        epoch_std_labels = []

        for batch_idx, (img_batch, tab_batch, label_batch) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                img_batch = img_batch.to(device)
                tab_batch = tab_batch.to(device)
                label_batch = label_batch.to(device)

                outs = model(img_batch, tab_batch)

                outs_mean, outs_std = outs[:, :, 0], outs[:, :, 1]
                label_mean, label_std = label_batch[:, :, 0], label_batch[:, :, 1]

                loss_mean = (criterion_mean(outs_mean, label_mean).mean(dim=1)).mean()
                loss_std = (criterion_std(outs_std, label_std).mean(dim=1)).mean()
                loss = loss_mean + loss_std

            val_losses_mean.append(loss_mean.detach().cpu().item())
            val_losses_std.append(loss_std.detach().cpu().item())
            tqdm.write(f'Val Epoch {epoch} | MSE Mean {val_losses_mean[-1]:.4g} | MSE STD {val_losses_std[-1]:.4g}')

            if args.use_wandb:
                wandb.log({
                    "val_losses_mean": val_losses_mean[-1], 
                    "val_losses_std": val_losses_std[-1],
                    "epoch":  epoch + batch_idx / n_batches_per_val_epoch,
                })
            
            # Yes Label Normalization
            label_mean_denorm = denormalize(label_mean, norm_params['speed_mean']['mean'], norm_params['speed_mean']['std']).detach().cpu().tolist()
            label_std_denorm = denormalize(label_std, norm_params['speed_std']['mean'], norm_params['speed_std']['std']).detach().cpu().tolist()
            outs_mean_denorm = denormalize(outs_mean, norm_params['speed_mean']['mean'], norm_params['speed_mean']['std']).detach().cpu().tolist()
            outs_std_denorm = denormalize(outs_std, norm_params['speed_std']['mean'], norm_params['speed_std']['std']).detach().cpu().tolist()

            epoch_mean_labels.extend(label_mean_denorm)
            epoch_std_labels.extend(label_std_denorm)
            epoch_mean_preds.extend(outs_mean_denorm)
            epoch_std_preds.extend(outs_std_denorm)

        epoch_mean_preds = np.array(epoch_mean_preds)
        epoch_std_preds = np.array(epoch_std_preds)
        epoch_mean_labels = np.array(epoch_mean_labels)
        epoch_std_labels = np.array(epoch_std_labels)

        epoch_mean_preds = epoch_mean_preds.flatten()
        epoch_std_preds = epoch_std_preds.flatten()
        epoch_mean_labels = epoch_mean_labels.flatten()
        epoch_std_labels = epoch_std_labels.flatten()

        epoch_mean_rmse = np.sqrt(np.mean(((epoch_mean_preds - epoch_mean_labels) ** 2)))
        epoch_std_rmse = np.sqrt(np.mean(((epoch_std_preds - epoch_std_labels) ** 2)))
        epoch_mean_mae = mean_absolute_error(epoch_mean_labels, epoch_mean_preds)
        epoch_std_mae = mean_absolute_error(epoch_std_labels, epoch_std_preds)
        epoch_mean_r2 = r2_score(epoch_mean_labels, epoch_mean_preds)
        epoch_std_r2 = r2_score(epoch_std_labels, epoch_std_preds)

        rmse_mean_val_list.append(epoch_mean_rmse )
        rmse_std_val_list.append(epoch_std_rmse)
        mae_mean_val_list.append(epoch_mean_mae)
        mae_std_val_list.append(epoch_std_mae)
        r2_mean_val_list.append(epoch_mean_r2)
        r2_std_val_list.append(epoch_std_r2)

        print((
            f'\nEpoch Val Mean RMSE {rmse_mean_val_list[-1]:.4g}, '
            f'MAE {mae_mean_val_list[-1]:.4g}, '
            f'R2 {r2_mean_val_list[-1]:.4g}'
        ))
        print((
            f'Epoch Val STD RMSE {rmse_std_val_list[-1]:.4g}, '
            f'MAE {mae_std_val_list[-1]:.4g}, '
            f'R2 {r2_std_val_list[-1]:.4g}\n'
        ))
        if args.use_wandb:
            wandb.log({
                "val_mean_rmse": rmse_mean_val_list[-1],
                "val_std_rmse": rmse_std_val_list[-1],
                "val_mean_mae": mae_mean_val_list[-1],
                "val_std_mae": mae_std_val_list[-1],
                "val_mean_r2": r2_mean_val_list[-1],
                "val_std_r2": r2_std_val_list[-1],
                "epoch": epoch,
            })
        
        if rmse_mean_val_list[-1] < best_mean_rmse:
            best_mean_rmse = rmse_mean_val_list[-1]
            try:
                print(f"Best Mean Speed RMSE {best_mean_rmse} found. Saving model...")
                torch.save(model.state_dict(), os.path.join(args.model_save_dir,
                    f'best-{run_name}.pth'))
            except Exception as e:
                print(f"Failed to save the model due to error: {e}")
    
    try:
        torch.save(model.state_dict(), os.path.join(args.model_save_dir,
            f'{run_name}.pth'))
    except Exception as e:
        print(f"Failed to save the model due to error: {e}")
    
    if args.use_wandb:
        wandb.finish()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--hidden_dim', type=int, default=64)
    
    parser.add_argument('--norm_params_file', type=str, default='crop_info_norm_params.yaml')
    parser.add_argument('--tabular_cols_file', type=str, default='t_cols_norm.yaml')
    parser.add_argument('--csv_path', type=str, default='data/crop_info_full_4.csv')
    parser.add_argument('--image_cols', nargs='+', default=['rgb_path', 'nir_path'])
    
    parser.add_argument('--model_save_dir', type=str, default='results/models')
    parser.add_argument('--use_wandb', action='store_true', default=False)

    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--rnn_type', type=str, default='GRU')
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--no_std', action='store_true', default=False)
    parser.add_argument('--freeze_enc', action='store_true', default=False)
    
    args = parser.parse_args()

    main(args)