import json
import os
import time
import sys
sys.path.append(os.path.abspath(".."))
from argparse import ArgumentParser
from functools import partial
import numpy as np
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import nets.losses as losses
from src import dataloading, summary_utils, utils, models
import random
import scipy.io as sio
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

utils.seed(123)
torch.set_num_threads(2)
device = 'cuda:0'

def main(args):
    if 'lsvpe' not in args.shutter:
        print('SETTING MLR TO 2E-4 since we are only training decoder')
        args.mlr = 2e-4
    args.slr = float(args.slr)
    args.mlr = float(args.mlr)

    if args.exp_name == '':
        response = input('You did not specify an exp_name, want to do that now? (type name or "N") : ')
        if response.lower() != 'n':
            args.exp_name = response.lower()

    lpips_fn = losses.LPIPS_1D()
    args, dir_name,restore_name = utils.modify_args(args)

    # define model
    model_dir, version_num = utils.make_model_dir(dir_name, args.test, args.exp_name)

    shutter = models.define_shutter(args.shutter, args, model_dir=model_dir)
    decoder = models.define_decoder(args.decoder, args)
    model = models.define_model(shutter, decoder, args, get_coded=False)

    model.train()
    model.cuda()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Num trainable params: {params}')

    summaries_dir, checkpoints_dir = utils.make_subdirs(model_dir)
    writer = SummaryWriter(summaries_dir)
  #   optim = torch.optim.Adam([
   #      {'params': model.shutter.parameters(), 'lr': args.slr},
   #      {'params': model.decoder.parameters(), 'lr': args.mlr}], lr=args.mlr, betas=(0.9, 0.999))
    optim = torch.optim.Adam([
        # {'params': model.shutter.parameters(), 'lr': args.slr},eps=1e-2
        {'params': model.decoder.parameters(), 'lr': args.mlr}
        ], lr=args.mlr, betas=(0.9, 0.999))

    optim1 = torch.optim.SGD([
        {'params': model.shutter.parameters(), 'lr': args.slr}], lr=args.mlr, momentum=0.9)

    if args.date_resume != '00-00-00':
        print('Loading checkpoint')
        file = [f for f in os.listdir(f'{restore_name}/v_{4}/checkpoints') if 'model_epoch_0133.pth' in f][0]
        fname = f'{restore_name}/v_{4}/checkpoints/{file}'
        # file = [f for f in os.listdir(f'{restore_name}/v_{42}/checkpoints') if 'model_epoch_0120.pth' in f][0]
        # fname = f'{restore_name}/v_{42}/checkpoints/{file}'
        print(fname)
        checkpoint = torch.load(fname)
        # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # optim.load_state_dict(checkpoint['optim_state_dict'])
        #  try:
        #    current_model_dict = model.state_dict()
        # enabling this if its for training
        # print(checkpoint['model_state_dict'].keys())

        #For continual training
        if 'adaptive' in restore_name:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optim.load_state_dict(checkpoint['optim_state_dict'])
        else:
            new_state_dict = {k: v for k, v in
                              zip(checkpoint['model_state_dict'].keys(), checkpoint['model_state_dict'].values()) if
                              'shutter.model.body' not in k}
            model.load_state_dict(new_state_dict, strict=False)



        #        new_state_dict={k:v for k,v in zip(checkpoint['optim_state_dict'].keys(), checkpoint['optim_state_dict'].values()) if 'shutter.model.body.2' not in k}
        #       optim.load_state_dict(new_state_dict)
        # some baselines won't have optim state dict, doesn't affect training much  shutter.model.body.2.body.4
        # new_state_dict = {k: v for k, v in
        #                   zip(checkpoint['model_state_dict'].keys(), checkpoint['model_state_dict'].values()) if
        #                   'shutter.model.body' not in k}
        # model.load_state_dict(new_state_dict, strict=False)

        # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    #     optim.load_state_dict(checkpoint['optim_state_dict'])
    #     optim1.load_state_dict(checkpoint['optim1_state_dict'])
    #    except KeyError:
    #   current_model_dict = model.state_dict()
    # new_state_dict={k:v for k,v in zip(current_model_dict.keys(), current_model_dict.values()) if 'shutter.model.body.4.weight' not in k}
    #        model.load_state_dict(checkpoint, strict=False)
    else:
        if not args.test:
            with open(f'{dir_name}/args.json', 'w') as f:
                json.dump(vars(args), f, indent=4)

    train_dataloader = dataloading.loadTrainingDataset(args)
    val_dataloader, val_dataset = dataloading.loadValDataset(args)
    summary_fn = partial(summary_utils.write_summary, args.batch_size, writer, args.shutter)

    scheduler = utils.define_schedule(optim)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optim1, milestones=[50, 100, 150, 200, 250], gamma=0.5)

    loss_fn = utils.define_loss(args)

    total_time_start = time.time()
    best_val_psnr = 0
    total_steps = 0

    f = open('progress.txt', "w")
    f.write("I have deleted the content!")
    f.close()

    with tqdm(total=len(train_dataloader) * args.max_epochs) as pbar:
        for epoch in range(args.max_epochs):
            print(epoch)
            for step, (avg, model_input, gt, ref_input, vid_num, clip_num) in enumerate(train_dataloader):

                model_input = model_input.cuda()
                gt = gt.cuda()
                ref_input = ref_input.cuda()
                start_time = time.time()

                optim.zero_grad(set_to_none=True)
                optim1.zero_grad(set_to_none=True)

                train_losses = []
                # for itr in range(1):
                if epoch<120:
                    restored, coded, actions, log_prob = model([model_input, ref_input], train=True, steps=total_steps)
                else:
                    restored, coded, actions, log_prob = model([model_input, ref_input], train=False, steps=total_steps)

                actions = actions.float()
                y = torch.zeros_like(actions)

                    #  y = torch.where(actions == -1, 0*torch.ones_like(actions), y)
                y = torch.where(actions == 0, 1 * torch.ones_like(actions), y)
                y = torch.where(actions == 1, 2 * torch.ones_like(actions), y)
                y = torch.where(actions == 2, 4 * torch.ones_like(actions), y)
                y = torch.where(actions == 3, 8 * torch.ones_like(actions), y)

                    #      train_loss1=0

                train_loss1 = (restored - gt).pow(2).mean(dim=[1, 2, 3])

                    #    if log_prob is None:
                    #        train_loss = train_loss1
                    #    else:
                    #        rewards=(torch.log(1./((restored - gt).pow(2).mean(dim=[1,2,3]).detach()+1e-6)))

                    #        train_loss = train_loss1-(rewards-10.*y.mean(dim=[1,2,3]).detach())*(log_prob.mean(dim=[1,2,3]))
                if log_prob is None:
                    train_loss = train_loss1
                    train_losses.append(train_loss)
                else:
                    rewards = (torch.log(1. / ((restored - gt).pow(2).detach() + 1e-4)))

                    train_loss = train_loss1 - ((rewards - 0.05 * y.detach()) * log_prob).mean()

                    train_losses.append(train_loss)
                # train_loss2= (coded[:,:-1,:,:]-gt).pow(2).mean()-4.*actions.detach()  -0.1*train_loss2.mean() +-0.01*y.detach()  0.008   0.012 0.03
                # gamma 1. Action 1.817
                # print(train_loss)
                train_loss = torch.concat(train_losses, dim=0).mean()

                train_loss.backward()
                # v=model.shutter.model[-1][-2].weight.grad.detach().cpu().numpy()
                # print(np.mean(np.abs(v)))
                # v = model.shutter.end_params.grad.detach().cpu().numpy()
                # print(np.mean(np.abs(v)))
                optim.step()
                optim1.step()
                pbar.update(1)
                if not total_steps % args.steps_til_summary and total_steps > 50:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)
                    # summary_fn(model, [model_input[:1], ref_input[:1]], gt[:1, :, :, :], restored[:1, :, :, :], avg,
                    #            total_steps, optim)
                    tqdm.write("Epoch %d, Total loss %0.6f, "
                               "iteration time %0.6f s, total time %0.6f min" % (
                                   epoch, train_loss, time.time() - start_time,
                                   (time.time() - total_time_start) / 60.0))

                    if val_dataloader is not None:
                        with torch.no_grad():
                            model.eval()
                            val_losses = []
                            val_psnrs = []
                            val_lpips = []
                            val_ssims = []
                            val_actions = []
                            for (avg, model_input, gt, ref_input, vid_num, clip_num) in tqdm(val_dataloader, disable=True):
                                model_input = model_input.cuda()
                                gt = gt.cuda()
                                ref_input = ref_input.cuda()
                                restored, coded, actions, rate_loss = model([model_input, ref_input], train=False)
                                #  restored=restored.reshape([-1,5,16,args.block_size[1],args.block_size[2]])
                                #   restored=restored[-1]

                                val_loss = loss_fn(restored, gt)
                                psnr = summary_utils.get_psnr(restored, gt)
                                ssim = summary_utils.get_ssim(restored, gt)
                                # vlpips = lpips_fn(restored, gt)
                                # actions=torch.round(actions)
                                actions = actions.float()
                                y = torch.zeros_like(actions)

                                #  y = torch.where(actions == -1, 0*torch.ones_like(actions), y)
                                y = torch.where(actions == 0, 1 * torch.ones_like(actions), y)
                                y = torch.where(actions == 1, 2 * torch.ones_like(actions), y)
                                y = torch.where(actions == 2, 4 * torch.ones_like(actions), y)
                                y = torch.where(actions == 3, 8 * torch.ones_like(actions), y)

                                val_ssims.append(ssim)
                                val_psnrs.append(psnr)
                                val_losses.append(val_loss.item())
                                val_lpips.append(0)
                                val_actions.append(y.mean().detach().cpu().numpy())
                                # print(f'PSNR: '
                                #       f'{np.mean(val_psnrs)}, SSIM: {np.mean(val_ssims)}, LPIPS: {np.mean(val_lpips)}, Action:{np.mean(val_actions)}')

                            # summary_utils.write_val_scalars(writer,
                            #                                 ['psnr', 'ssim', 'lpips', 'loss'],
                            #                                 [val_psnrs, val_ssims, val_lpips, val_losses],
                            #                                 total_steps)
                            utils.save_chkpt(model, optim, optim1, checkpoints_dir, epoch=epoch, final=False)

                            # if np.mean(val_psnrs) > best_val_psnr:
                            print(f'PSNR: '
                                  f'{np.mean(val_psnrs)}, SSIM: {np.mean(val_ssims)}, LPIPS: {np.mean(val_lpips)}, Action:{np.mean(val_actions)}')
                            #  best_val_psnr = np.mean(val_psnrs)
                            f = open('progress.txt' , "a")
                            f.write(f'#Test iteration %d, ,'
                                    f'PSNR %.8f, SSIM %.8f, LPIPS %.8f, Action %.8f #\n' % (
                                    epoch,
                                    np.mean(val_psnrs),
                                    np.mean(val_ssims),
                                    np.mean(val_lpips),
                                    np.mean(val_actions)
                                    ))
                            f.close()

                            utils.save_chkpt(model, optim, optim1, checkpoints_dir, epoch=epoch, best=True)
                            utils.save_best_metrics(args, total_steps, epoch, val_psnrs,
                                                    val_ssims, val_lpips, checkpoints_dir)
                        model.train()
                        scheduler.step(val_loss)
                        scheduler1.step()
                total_steps += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root',
                        type=str,
                        default='../data')
    parser.add_argument('--log_root',
                        type=str,
                        default='../logs')
    parser.add_argument('--test', action='store_true',
                        help='dummy experiment name for just testing code')
    parser.add_argument('-b', '--block_size',
                        help='delimited list input for block size in format %,%,%',
                        default='8,256,256')
    parser.add_argument('--resume',
                        type=str,
                        default='24-03-08',
                        help='date of folder of exp to resume')
    parser.add_argument('--gt', type=int, default=0)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--scale', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='MST_adaptive')
    parser.add_argument('--restore_dic_name', type=str, default='MST_fixed')
    parser.add_argument('--reg', type=float, default=100.0, help='regularization on lpips loss')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--max_epochs', type=int, default=300)

    parser.add_argument('--mlr', help='model_lr', type=str, default='5e-5')
    parser.add_argument('--slr', help='shutter_lr', type=str, default='5e-3')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--steps_til_summary', type=int, default=500)
    parser.add_argument('--epochs_til_checkpoint', type=int, default=1000)
    parser.add_argument('--steps_til_ckpt', type=int, default=5000)
    parser.add_argument('--interp', type=str, default='none',
                        choices=['none', 'bilinear', 'scatter'])
    parser.add_argument('--init', type=str, choices=['even', 'ones', 'quad'], default='quad',
                        help='choose way to initialize learned shutters, '
                             'even=even probabilities on all options,'
                             'ones=use all ones,'
                             'quad=use quad structure')

    parser.add_argument('--loss', type=str, choices=['mpr', 'l1', 'l2_lpips', 'l2'], default='l2')
    parser.add_argument('--decoder', type=str,
                        choices=['unet', 'mpr', 'dncnn',],
                        default='MST')
    parser.add_argument('--shutter', type=str, default='lsvpe')
    parser.add_argument('--sched', type=str, default='reduce')

    args = parser.parse_args()
    args.block_size = [int(item) for item in args.block_size.split(',')]
    main(args)