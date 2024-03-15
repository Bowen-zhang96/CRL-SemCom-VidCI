import torch
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def get_psnr(pred, gt):
    return 10 * torch.log10(1 / torch.mean((pred - gt) ** 2)).detach().cpu().numpy()


def get_ssim(pred, gt):
    ssims = []
    for i in range(pred.shape[0]):
        pred_i = pred[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        gt_i = gt[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        ssims.append(ssim(pred_i, gt_i, data_range=1., channel_axis=2))
    return sum(ssims) / len(ssims)


def write_val_scalars(writer, names, values, total_steps):
    for name, val in zip(names, values):
        writer.add_scalar(f'val/{name}', np.mean(val), total_steps)


def write_summary(batch_size, writer, shutter_name, model, input, gt,
                  output, avg, total_steps, optim):
    coded, actions, rate_loss = model.shutter(input)
    # coded=coded1[:,0,:,:]

    cat_input = coded[:, 0, :, :].unsqueeze(0)
    for i in range(1, coded.shape[1]):
        cat_input = torch.cat((cat_input, coded[:, i, ...].unsqueeze(0)), dim=0)

    grid = make_grid(cat_input,
                     scale_each=True, nrow=1, normalize=False).cpu().detach().numpy()
    writer.add_image(f"sensor images", grid, total_steps)

    # output_c=output.cpu()
    cat_input = output[:, 0, :, :].unsqueeze(0)
    for i in range(1, output.shape[1]):
        cat_input = torch.cat((cat_input, output[:, i, ...].unsqueeze(0)), dim=0)

    grid = make_grid(cat_input,
                     scale_each=True, nrow=1, normalize=False).cpu().detach().numpy()
    writer.add_image(f"output images", grid, total_steps)

    # gt = gt.cpu()
    cat_input = gt[:, 0, :, :].unsqueeze(0)
    for i in range(1, output.shape[1]):
        cat_input = torch.cat((cat_input, gt[:, i, ...].unsqueeze(0)), dim=0)

    grid = make_grid(cat_input,
                     scale_each=True, nrow=1, normalize=False).cpu().detach().numpy()
    writer.add_image(f"output images", grid, total_steps)

    psnr_=[]
    ssim_=[]
    for i in range(0, output.shape[1]):
        psnr_.append(get_psnr(output[:, i:i+1, ...], gt[:, i:i+1, ...]))
        ssim_.append(get_ssim(output[:, i:i+1, ...], gt[:, i:i+1, ...]))
    psnr = np.mean(np.asarray(psnr_))
    ssim = np.mean(np.asarray(ssim_))
    writer.add_scalar(f"train/psnr", psnr, total_steps)
    writer.add_scalar(f"train/ssim", ssim, total_steps)
    writer.add_scalar("learning_rate", optim.param_groups[0]['lr'], total_steps)

    if 'lsvpe' in shutter_name:
        fig = plt.figure()
        plt.bar(model.shutter.counts.keys(), model.shutter.counts.values())
        plt.ylabel('counts')
        writer.add_figure(f'lengths_freq', fig, total_steps)

        shutter = model.shutter.lengths.detach().cpu()

        fig = plt.figure()
        plt.imshow(shutter[0,0,:,:])
        plt.colorbar()
        writer.add_figure(f'train/learned_length', fig, total_steps)
