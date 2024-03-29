import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lpips


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=0.5, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class LPIPS_1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceptual_loss = lpips.LPIPS(net='vgg').cuda()

    def forward(self, y_pred, gt):
        y_pred = y_pred.repeat(1, 3, 1, 1)
        gt = gt.repeat(1, 3, 1, 1)
        return torch.mean(self.perceptual_loss.forward(y_pred, gt)).detach().cpu().item()


class MPRNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_loss = EdgeLoss()
        self.char_loss = CharbonnierLoss()

    def forward(self, y_pred, gt):
        y_pred = y_pred.repeat(1, 3, 1, 1)
        gt = gt.repeat(1, 3, 1, 1)
        return self.char_loss(y_pred, gt) + (0.05 * self.edge_loss(y_pred, gt))


class L2LPIPSRegLoss(nn.Module):
    def __init__(self, reg):
        super().__init__()
        self.l2_loss = nn.MSELoss()
        self.perceptual_loss = lpips.LPIPS(net='vgg').cuda()
        self.reg = reg

    def forward(self, y_pred, gt):
        return self.l2_loss(y_pred, gt) + self.reg * self.perceptual_loss.forward(y_pred, gt).mean()