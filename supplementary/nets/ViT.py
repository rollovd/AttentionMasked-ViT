import math
import torch
import logging
import numpy as np
import torch.nn as nn
from functools import partial
from collections import OrderedDict

from supplementary.helpers.helpers import named_apply
from supplementary.nets.layers import PatchEmbed, Block, trunc_normal_, lecun_normal_

_logger = logging.getLogger(__name__)

class VisionTransformer(nn.Module):

    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1, embed_dim=192, depth=6,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', attn_mask=False, device=None, pos_embed='harmonic',
                 head_name='apart', sigma=96, key_bias=False, query_bias=False, value_bias=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.device = device
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.attn_mask = attn_mask
        self.query_bias = query_bias
        self.key_bias = key_bias
        self.value_bias = value_bias

        self.positional_emb_name = pos_embed

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self._init_pos_embed(num_patches, embed_dim, num_heads)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, device=device,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, attn_mask=attn_mask,
                query_bias=query_bias, key_bias=key_bias, value_bias=value_bias, num_patches=num_patches,
                rpe1d=self.positional_emb_name=='rpe1d')
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.init_weights(weight_init)

        self.head_name = head_name
        if self.head_name == 'apart':
            self.head = OutClassifierHead(sigma=sigma)

        elif self.head_name == 'vanilla':
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            self.sigmoid = nn.Sigmoid()

    def _init_pos_embed(self, num_patches, embed_dim, num_heads):
        self.pos_embed = None
        if self.positional_emb_name == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        elif self.positional_emb_name == 'harmonic':
            self.pos_embed = self.__create_harmonic_pe(emb_dim=embed_dim, num_patches=num_patches + self.num_tokens).to(
                self.device)

    @staticmethod
    def __create_harmonic_pe(emb_dim, num_patches):
        position_encoding = np.array([
            [pos / np.power(10000, 2 * i / emb_dim) for i in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(num_patches)])

        position_encoding[1:, 0::2] = np.sin(position_encoding[1:, 0::2])
        position_encoding[1:, 1::2] = np.cos(position_encoding[1:, 1::2])
        output = torch.from_numpy(position_encoding).type(torch.FloatTensor)

        return output

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if mode.startswith('jax'):
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def forward_features(self, x, **kwargs):
        keys = list(kwargs.keys())
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)

        if self.pos_embed is not None and self.positional_emb_name != 'rpe1d':
            x = self.pos_drop(x + self.pos_embed)
        else:
            x = self.pos_drop(x)

        for index, block in enumerate(self.blocks):
            x = block(x, attn_mask=kwargs['attn_mask']) if 'attn_mask' in keys else block(x)

        x = self.norm(x)

        return self.pre_logits(x[:, 0])

    def forward(self, x, **kwargs):
        keys = list(kwargs.keys())
        x = self.forward_features(x, attn_mask=kwargs['attn_mask']) if 'attn_mask' in keys else self.forward_features(x)
        x = self.head(x)

        if self.head_name == 'vanilla':
            x = self.sigmoid(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

class OutClassifierHead(nn.Module):

    def __init__(self, sigma, **kwargs):
        super(OutClassifierHead, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.sqrt(torch.sum(x*x, dim=-1))
        x = 1.0 - torch.exp(-x*x/self.sigma)
        return x

if __name__ == "__main__":
    vit = VisionTransformer(
        img_size=256,
        embed_dim=192
    )

    input = torch.randn(10, 3, 256, 256)
    output = vit(input)