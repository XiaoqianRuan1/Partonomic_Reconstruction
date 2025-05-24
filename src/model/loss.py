from math import exp
import torch
from torch import nn
import torch.nn.functional as F
import torchvision


def get_loss(name):
    return {
        'bce': nn.BCEWithLogitsLoss,
        'bce_iou': BceIouLoss,
        'mse': nn.MSELoss,
        'l2': nn.MSELoss,
        'l1': nn.L1Loss,
        'huber': nn.SmoothL1Loss,
        'cosine': nn.CosineSimilarity,
        'perceptual': PerceptualLoss,
        'ssim': SSIMLoss,
        'mask': IoULoss,
    }[name]


class PerceptualLoss(nn.Module):
    def __init__(self, normalize_input=True, normalize_features=True, feature_levels=None, sum_channels=False,
                 requires_grad=False):
        super().__init__()
        self.normalize_input = normalize_input
        self.normalize_features = normalize_features
        self.sum_channels = sum_channels
        self.feature_levels = feature_levels if feature_levels is not None else [3]
        assert isinstance(self.feature_levels, (list, tuple))
        self.max_level = max(self.feature_levels)
        self.register_buffer('mean_rgb', torch.Tensor([0.485, 0.456, 0.406]))
        self.register_buffer('std_rgb', torch.Tensor([0.229, 0.224, 0.225]))

        layers = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = layers[:4]     # relu1_2
        self.slice2 = layers[4:9]    # relu2_2
        self.slice3 = layers[9:16]   # relu3_3
        self.slice4 = layers[16:23]  # relu4_3
        self.slice5 = layers[23:30]  # relu5_3
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, im1, im2):
        inp = torch.cat([im1, im2], 0)
        if self.normalize_input:
            inp = (inp - self.mean_rgb.view(1, 3, 1, 1)) / self.std_rgb.view(1, 3, 1, 1)

        feats = []
        for k in range(1, 6):
            if k > self.max_level:
                break
            inp = getattr(self, f'slice{k}')(inp)
            feats.append(torch.chunk(inp, 2, dim=0))

        losses = []
        for k, (f1, f2) in enumerate(feats, start=1):
            if k in self.feature_levels:
                if self.normalize_features:
                    f1, f2 = map(lambda t: t / (t.norm(dim=1, keepdim=True) + 1e-10), [f1, f2])
                loss = (f1 - f2) ** 2
                if self.sum_channels:
                    losses.append(loss.sum(1).flatten(2).mean(2))
                else:
                    losses.append(loss.flatten(1).mean(1))
        return sum(losses)



######################################################################
# SSIM original repo implem: https://github.com/Po-Hsun-Su/pytorch-ssim
######################################################################


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()



class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self.create_window(window_size, channel)

    def ssim(self, img1, img2):
        window_size, channel = self.window_size, self.channel
        window = self.window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map

    @staticmethod
    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, inp, target):
        return self.ssim(inp, target).flatten(1).mean(1)

class IoULoss(torch.nn.Module):
    def __init__(self, eps = 1e-6, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        
    def iou(self, predict, target):
        dims = tuple(range(predict.ndimension())[1:])
        intersect = (predict * target).sum(dims)
        union = (predict + target - predict * target).sum(dims) + self.eps
        if self.reduction == 'none':
            return (intersect / union)
        return (intersect / union).sum() / intersect.nelement()

    def forward(self, predict, target):
        return 1 - self.iou(predict, target)
    

def mask_loss(val_gt, val_pred, reduction_method='sum', eps=0., sigma_pow=1, feat_dim=True):
    assert(val_gt.shape==val_pred.shape)
    loss_out = (val_gt - val_pred).abs()
    loss_out = (loss_out + eps).pow(sigma_pow)
    if feat_dim:
        loss_out = loss_out.sum(-1)
    return apply_reduction(loss_out,reduction_method)

def apply_reduction(tensor, reduction_method):
    if reduction_method == "sum":
        return tensor.sum()
    elif reduction_method == "mean":
        return tensor.mean()
        
class BceIouLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def bce_iou_loss(self, pred, mask, input_shape=[64, 64], batch_size=32):
        size = pred.size()[2:]
        epsilon=1e-7
        #mask = F.interpolate(mask, size=size, mode='bilinear')
        pred = torch.clamp(pred,epsilon,1.0-epsilon)
        wbce = F.binary_cross_entropy(pred, mask)
        # Compute predicted area and input area
        pred_area = torch.sum(pred)
        mask_area = torch.sum(mask)
        input_area = pred.size()[0] * input_shape[0] * input_shape[1]
        input_mask = pred.size()[0] * input_shape[0] * input_shape[1]
        inter = (pred * mask).sum(dim=(2, 3))
        union = (pred + mask).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou)

    def forward(self, pred, mask):
        return self.bce_iou_loss(pred,mask).flatten(1).mean(1)