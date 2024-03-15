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
    args, dir_name = utils.modify_args(args)

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
        file = [f for f in os.listdir(f'{dir_name}/v_{34}/checkpoints') if 'model_epoch_0006.pth' in f][0]
        fname = f'{dir_name}/v_{34}/checkpoints/{file}'
        print(fname)
        checkpoint = torch.load(fname)

    
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optim.load_state_dict(checkpoint['optim_state_dict'])
     

    else:
        if not args.test:
            with open(f'{dir_name}/args.json', 'w') as f:
                json.dump(vars(args), f, indent=4)

    train_dataloader = dataloading.loadTrainingDataset(args)
    val_dataloader, val_dataset = dataloading.loadValDataset(args)
    vid_dict=val_dataset.vid_dict
    summary_fn = partial(summary_utils.write_summary, args.batch_size, writer, args.shutter)


    loss_fn = utils.define_loss(args)


    
    model.eval()
    val_losses = []
    val_psnrs = []
    val_lpips = []
    val_ssims = []
    val_actions=[]
    
    restored_s=[]
    gt_s=[]
    actions_s=[]
    idx=0
    files_dict=[]
    for (avg, model_input, gt, ref_input, vid_num, clip_num) in tqdm(val_dataloader, disable=False):
        vid_num=vid_num.detach().cpu().numpy()
        clip_num=clip_num.detach().cpu().numpy()
        for tmp in range(vid_num.shape[0]):
            
            vid = vid_dict[vid_num[tmp]][clip_num[tmp]]
            
            
            vid=vid[8]
           
            files_dict.append(vid)
        idx=idx+1
    #     if idx>50:  args.batch_size
   #          break
        model_input = model_input.cuda()
        gt = gt.cuda()
        ref_input=ref_input.cuda()
        restored, coded, actions, rate_loss = model([model_input,ref_input], train=False)
        
        

        val_loss = loss_fn(restored, gt)
        psnr = summary_utils.get_psnr(restored, gt)
        ssim = summary_utils.get_ssim(restored, gt)
                                # vlpips = lpips_fn(restored, gt)
        actions=actions.float()
        y = torch.zeros_like(actions)

                              #  y = torch.where(actions == -1, 0*torch.ones_like(actions), y)
        y = torch.where(actions == 0, 1*torch.ones_like(actions), y)
        y = torch.where(actions == 1, 2*torch.ones_like(actions), y)
        y = torch.where(actions == 2, 4*torch.ones_like(actions), y)
        y = torch.where(actions == 3, 8*torch.ones_like(actions), y)
        y = torch.where(actions == 4, 16*torch.ones_like(actions), y)
                                

        val_ssims.append(ssim)
        val_psnrs.append(psnr)
        val_losses.append(val_loss.item())
        val_lpips.append(0)
        val_actions.append(y.mean().detach().cpu().numpy())
        
        restored_s.append(restored.detach().cpu().numpy())
        gt_s.append(gt.detach().cpu().numpy())
        # print(actions[0,0,:,:])
        actions_s.append(y.detach().cpu().numpy())

        # val_ssims.append(ssim)
        # val_psnrs.append(psnr)
        # val_losses.append(val_loss.item())
        # val_lpips.append(0)

       # summary_utils.write_val_scalars(writer, ['psnr', 'ssim', 'lpips', 'loss'],[val_psnrs, val_ssims, val_lpips, val_losses],
       # total_steps)

        
    print(f'BEST PSNR: 'f'{np.mean(val_psnrs)}, SSIM: {np.mean(val_ssims)}, Action:{np.mean(val_actions)}')
    
    restored_s=np.concatenate(restored_s,axis=0)
    gt_s=np.concatenate(gt_s,axis=0)
    actions_s=np.concatenate(actions_s,axis=0)
    idx=np.arange(len(restored_s))
    print(idx)
    np.random.shuffle(idx)
    idx=idx[:20]
    sio.savemat('results_adaptive_PSNR_%.5f_Action_%.5f.mat'%(np.mean(val_psnrs), np.mean(val_actions)),{'restored':restored_s[idx], 'gt':gt_s[idx],'action':actions_s[idx],'files':np.take(files_dict,idx)})
                                


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
    parser.add_argument('--reg', type=float, default=100.0, help='regularization on lpips loss')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--max_epochs', type=int, default=6000)

    parser.add_argument('--mlr', help='model_lr', type=str, default='5e-5')
    parser.add_argument('--slr', help='shutter_lr', type=str, default='5e-3')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--steps_til_summary', type=int, default=1000)
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