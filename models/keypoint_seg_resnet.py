import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models



class KeypointUpSample(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = nn.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        #nn.init.uniform_(self.kps_score_lowres.weight)
        #nn.init.uniform_(self.kps_score_lowres.bias)
        self.up_scale = 1
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        return torch.nn.functional.interpolate(
            x, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False, recompute_scale_factor=False
        )




class SpatialSoftArgmax(nn.Module):
    """
    The spatial softmax of each feature
    map is used to compute a weighted mean of the pixel
    locations, effectively performing a soft arg-max
    over the feature dimension.

    """

    def __init__(self, normalize=True):
        """Constructor.
        Args:
            normalize (bool): Whether to use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, h, device=device),
                    torch.linspace(-1, 1, w, device=device),
                    indexing='ij',
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, h, device=device),
                torch.arange(0, w, device=device),
                indexing='ij',
            )
        )

    def forward(self, x):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # compute a spatial softmax over the input:
        # given an input of shape (B, C, H, W),
        # reshape it to (B*C, H*W) then apply
        # the softmax operator over the last dimension
        b, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # create a meshgrid of pixel coordinates
        # both in the x and y axes
        yc, xc = self._coord_grid(h, w, x.device)

        # element-wise multiply the x and y coordinates
        # with the softmax, then sum over the h*w dimension
        # this effectively computes the weighted mean of x
        # and y locations
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)
        # print(f"softmax: {softmax.shape} xc flatten: {xc.flatten().shape} x_mean: {x_mean.shape}")
        # print(f"softmax: {softmax.shape} yc flatten: {yc.flatten().shape} y_mean: {y_mean.shape}")
        # print(x_mean[0])
        # torch.set_printoptions(profile="full")
        # print(f"top 5: {torch.topk(softmax, 5).values.sum(dim=1)}")
        # print(f"top 10: {torch.topk(softmax, 10).values.sum(dim=1)}")
        # print(f"top 15: {torch.topk(softmax, 15).values.sum(dim=1)}")
        # print(f"top 20: {torch.topk(softmax, 20).values.sum(dim=1)}")
        # print(f"top 30: {torch.topk(softmax, 30).values.sum(dim=1)}")
        # print(f"top 40: {torch.topk(softmax, 40).values.sum(dim=1)}")
        
        confidence = torch.topk(softmax, 40).values.sum(dim=1)
        # closest_indices = torch.topk(torch.abs(xc.flatten() - x_mean), 40, dim=1, largest=False).indices
        # y_closest_indices = torch.argmin(torch.abs(yc.flatten() - y_mean), dim=1).unsqueeze(dim=1)
        # print(closest_indices[0].shape, softmax[0].shape)
        # print(torch.sum(softmax[0].gather(0, closest_indices[0])))
        # print(softmax.gather(1, closest_indices))
        # print(torch.sum(softmax[0][closest_indices[0]-10: closest_indices[0]+10]))
        # print(torch.max(softmax, dim=1))
        # print(torch.argmin(torch.abs(xc.flatten() - x_mean)))

        # concatenate and reshape the result
        # to (B, C, 2) where for every feature
        # we have the expected x and y pixel
        # locations
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c, 2), confidence


class KeyPointSegNet(nn.Module):
    def __init__(self, args, lim=[-1., 1., -1., 1.], use_gpu=True):
        super(KeyPointSegNet, self).__init__()

        self.args = args
        self.lim = lim

        k = args.n_kp

        if use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"


        deeplabv3_resnet50 = models.segmentation.deeplabv3_resnet50(pretrained=True)
        deeplabv3_resnet50.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes

        self.backbone = torch.nn.Sequential(list(deeplabv3_resnet50.children())[0])

        self.read_out = KeypointUpSample(2048, k)

        self.spatialsoftargmax = SpatialSoftArgmax()

        self.classifer = torch.nn.Sequential((list(deeplabv3_resnet50.children())[1]))



    def forward(self, img):
        input_shape = img.shape[-2:]

        resnet_out = self.backbone(img)['out']  # (B, 2048, H//8, W//8)

        # keypoint prediction branch
        heatmap = self.read_out(resnet_out) # (B, k, H//4, W//4)
        keypoints, confidence = self.spatialsoftargmax(heatmap)
        # mapping back to original resolution from [-1,1]
        offset = torch.tensor([self.lim[0], self.lim[2]], device = resnet_out.device)
        scale = torch.tensor([self.args.width // 2, self.args.height // 2], device = resnet_out.device)
        keypoints = keypoints - offset
        keypoints = keypoints * scale

        # segmentation branch
        x = self.classifer(resnet_out)
        segout = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return keypoints, segout, confidence
