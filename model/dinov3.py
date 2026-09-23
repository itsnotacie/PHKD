"""DINOv3 backbone wrapper (via HF transformers), drop-in for model.dino.DINO.

Returns multilayer dense-cls features [B, 2*hidden, h, w] matching the DINO wrapper,
so the existing DPT localization head can consume DINOv3 features unchanged.
DINOv3 token order is [CLS, register tokens..., patch tokens...]; patches are taken
from the tail, CLS from index 0 (registers ignored), same as the DINOv2 wrapper.
"""
import torch

from transformers import AutoModel

from model.dino import center_padding, tokens_to_output

_HF_NAMES = {
    ("vitb16", "web"): "facebook/dinov3-vitb16-pretrain-lvd1689m",
    ("vitl16", "web"): "facebook/dinov3-vitl16-pretrain-lvd1689m",
    ("vitl16", "sat"): "facebook/dinov3-vitl16-pretrain-sat493m",
}

# Pretraining normalization (ImageNet for web LVD-1689M, satellite stats for SAT-493M).
_NORM = {
    "web": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "sat": ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143)),
}


class DINOv3(torch.nn.Module):
    def __init__(self, model_name="vitb16", variant="web", output="dense-cls", return_multilayer=True):
        super().__init__()
        key = (model_name, variant)
        if key not in _HF_NAMES:
            raise ValueError(f"Unsupported DINOv3 (model={model_name}, variant={variant}).")
        self.vit = AutoModel.from_pretrained(_HF_NAMES[key], output_hidden_states=True).eval().to(torch.float32)
        for param in self.vit.parameters():
            param.requires_grad = False

        mean, std = _NORM[variant]
        self.register_buffer("norm_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(std).view(1, 3, 1, 1))

        self.output = output
        self.patch_size = self.vit.config.patch_size
        hidden = self.vit.config.hidden_size
        feat_dim = hidden * 2 if output == "dense-cls" else hidden

        num_layers = self.vit.config.num_hidden_layers
        multilayers = [num_layers // 4 - 1, num_layers // 2 - 1, num_layers // 4 * 3 - 1, num_layers - 1]
        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            self.multilayers = [multilayers[-1]]
        self.layer = "-".join(str(x) for x in self.multilayers)

    def forward(self, images, branch=None):
        # incoming images are in [-1, 1]; convert to [0, 1] then apply pretraining norm
        images = (images + 1.0) / 2.0
        images = (images - self.norm_mean) / self.norm_std
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size
        num_spatial = h * w

        hidden_states = self.vit(pixel_values=images).hidden_states  # len = num_layers + 1
        outputs = []
        for layer in self.multilayers:
            x_i = hidden_states[layer + 1]  # +1: index 0 is the embedding output
            cls_tok = x_i[:, 0]
            spatial = x_i[:, -num_spatial:]
            outputs.append(tokens_to_output(self.output, spatial, cls_tok, (h, w)))
        return outputs[0] if len(outputs) == 1 else outputs


class DualDINOv3(torch.nn.Module):
    """Route satellite branch to the SAT-493M variant, ground branch to the web variant."""

    def __init__(self, arch="vitl16", output="dense-cls"):
        super().__init__()
        self.sat_backbone = DINOv3(arch, variant="sat", output=output)
        self.grd_backbone = DINOv3(arch, variant="web", output=output)
        self.feat_dim = self.sat_backbone.feat_dim
        self.patch_size = self.sat_backbone.patch_size

    def forward(self, images, branch="grd"):
        backbone = self.sat_backbone if branch == "sat" else self.grd_backbone
        return backbone(images)

