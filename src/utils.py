import torch, os, lpips
import numpy as np
import random
import datetime
import pandas as pd
import torch.nn as nn
import nets.losses as losses


def seed(num):
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)


def modify_args(args):
    if args.decoder == 'unet' and (args.loss != 'l2_lpips' and args.loss != 'l2'):
        raise Exception('unet + loss do not match')
    if 'lsvpe' in args.shutter and (args.interp != 'scatter' and args.interp != 'none'):
        raise Exception('learn all should use scatter or none')

    args.date_resume = args.resume
    if args.interp == 'none':
        args.interp = None
    # define folders
    if args.date_resume != '00-00-00':
        date = f'{args.date_resume}'
    else:
        date = datetime.date.today().strftime('%y-%m-%d')

    if args.test:
        exp_name = f'{args.log_root}/test'
        args.steps_til_summary = 10
    else:
        dir_name = f'{args.log_root}/{date}/{date}-{args.decoder}'
        os.makedirs(dir_name, exist_ok=True)

        printed_args = get_exp_name(args)
        if args.exp_name != '':
            exp_name = f'{dir_name}/{args.exp_name}'
        else:
            exp_name = f'{dir_name}/{printed_args}'

        if args.restore_dic_name != '':
            restore_name = f'{dir_name}/{args.restore_dic_name}'
        else:
            restore_name = f'{dir_name}/{printed_args}'

        print(exp_name)
        print(restore_name)
        if args.date_resume != '00-00-00' and not os.path.exists(exp_name):
            raise ValueError('This directory does not exist :-(')
    return args, exp_name, restore_name


def save_best_metrics(args, total_steps, epoch, val_psnrs, val_ssims, val_lpips, checkpoints_dir):
    single_column_names = ['Model', 'Shutter', 'Total Steps', 'Epoch', 'PSNR', 'SSIM', 'LPIPS']
    df = pd.DataFrame(columns=single_column_names)
    series = pd.Series([args.decoder,
                        args.shutter,
                        total_steps,
                        epoch,
                        round(np.mean(val_psnrs), 3),
                        round(np.mean(val_ssims), 3),
                        round(np.mean(val_lpips), 3)],
                       index=df.columns)
    # df = df.append(series, ignore_index=True)
    # df = pd.concat([df, series], ignore_index=True){'params': model.decoder.parameters(), 'lr': args.mlr}
    df.loc[len(df)]=series

    file_name = f'{checkpoints_dir}/val_results.csv'
    # overwrite the file and just save the best recent
    df.to_csv(file_name, header=single_column_names)


def define_loss(args):
    loss_dict = {'mpr': losses.MPRNetLoss(),
                 'l2_lpips': losses.L2LPIPSRegLoss(args.reg),
                 'l1': nn.L1Loss(),
                 'l2': nn.MSELoss()}
    return loss_dict[args.loss]


def make_model_dir(dir_name, test=False, exp_name=''):
    if test and exp_name != '':
        version_num = 0
        model_dir = f'{dir_name}/{exp_name}'
    else:
        version_num = find_version_number(dir_name)
        model_dir = f'{dir_name}/v_{version_num}'

    print(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir, version_num


def define_schedule(optim):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min',
                                                      factor=0.8,
                                                      patience=30,
                                                      threshold=1e-3,
                                                      threshold_mode='rel',
                                                      cooldown=0,
                                                      min_lr=5e-6,
                                                      eps=1e-08,
                                                      verbose=False)

# {'params': model.decoder.parameters(), 'lr': args.mlr}
def define_optim(model, args):
    if 'lsvpe' in args.shutter:
        # for name, param in model.decoder.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        # print(list(model.shutter.model.parameters()))
        # print(list(model.shutter.parameters()))
        optim = torch.optim.AdamW([
                                   {'params': model.shutter.parameters(), 'lr': args.slr},
                                   {'params': model.decoder.parameters(), 'lr': args.mlr}], lr=args.mlr, eps=1e-2)

    else:
        optim = torch.optim.AdamW([{'params': model.parameters(), 'lr': args.mlr}], lr=args.mlr)
    return optim


def save_chkpt(model, optim, optim1, checkpoints_dir, epoch=0, val_psnrs=None, final=False, best=False):
    if best:
        torch.save({'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'optim1_state_dict': optim1.state_dict()},
                   os.path.join(checkpoints_dir, 'model_best.pth'))
        return
    if final:
        torch.save({'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'optim1_state_dict': optim1.state_dict()},
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        return
    else:
        torch.save({'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'optim1_state_dict': optim1.state_dict()},
                   os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))

        if val_psnrs is not None:
            np.savetxt(os.path.join(checkpoints_dir, 'train_psnrs_epoch_%04d.txt' % epoch),
                       np.array(val_psnrs))
        return


def convert(img, dim=1):
    if dim == 1:
        return img.squeeze(0).squeeze(0).detach().cpu().numpy()
    if dim == 3:
        return img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


### functions for keeping train script clean ####
def make_subdirs(model_dir, make_dirs=True):
    summaries_dir = f'{model_dir}/summaries'
    checkpoints_dir = f'{model_dir}/checkpoints'
    if make_dirs:
        os.makedirs(summaries_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
    return summaries_dir, checkpoints_dir


def find_version_number(path):
    if not os.path.isdir(path):
        return 0
    fnames = os.listdir(path)
    print(fnames)
    # latest_s=''
    latest_v=0
    for i in range(len(fnames)):
        if 'args' in fnames[i] or 'ipynb' in fnames[i]:
            continue
        latest = fnames[i]
        latest = latest.rsplit('_', 1)[-1]
        latest=int(latest)
        if latest>latest_v:
            latest_v=latest
    return int(latest_v) + 1


def get_exp_name(args):
    ''' Make folder name readable '''
    printedargs = ''
    forbidden = ['data_root', 'log_root', 'test',
                 'remote', 'max_epochs', 'num_workers',
                 'epochs_til_checkpoint', 'date_resume',
                 'steps_til_ckpt', 'restart', 'slr', 'resume',
                  'gt', 'local', 'exp_name',]
    for k, v in vars(args).items():
        if k not in forbidden:
            print(f'{k} = {v}')
            if k == 'sched' and v == 'no_sched':
                continue
            if k == 'decoder':
                k = 'dec'
            if k == 'shutter':
                k = 'shut'
            printedargs += f'{k}={v}_'
    return printedargs


def augmentData(imgs):
    aug = random.randint(0, 8)
    num = len(imgs)
    # Data Augmentations
    if aug == 1:
        for i in range(num):
            imgs[i] = imgs[i].flip(-2)
    elif aug == 2:
        for i in range(num):
            imgs[i] = imgs[i].flip(-1)
    elif aug == 3:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i], dims=(-2, -1))
    elif aug == 4:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i], dims=(-2, -1), k=2)
    elif aug == 5:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i], dims=(-2, -1), k=3)
    elif aug == 6:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i].flip(-2), dims=(-2, -1))
    elif aug == 7:
        for i in range(num):
            imgs[i] = torch.rot90(imgs[i].flip(-1), dims=(-2, -1))
    return imgs
