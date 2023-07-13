# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange # rearrange用于将tensor的维度换位
import math

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock, Adapter


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        args,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        参数:
            img_size (int): 输入图像大小(默认为1024)
            patch_size (int): patch大小(默认为16)
            in_chans (int): 输入图像通道数(默认为3)
            embed_dim (int): patch嵌入维度(默认为768)
            depth (int): ViT深度(默认为12)
            mlp_ratio (float): mlp隐藏维度与嵌入维度的比例(默认为4.0)
            qkv_bias (bool): 如果为True,则在query,key,value上添加可学习的偏置(默认为True)
            norm_layer (nn.Module): 归一化层(默认为nn.LayerNorm)
            act_layer (nn.Module): 激活层(默认为nn.GELU)
            use_abs_pos (bool): 如果为True,则使用绝对位置嵌入(默认为True)
            use_rel_pos (bool): 如果为True,则在注意力图中添加相对位置嵌入(默认为False)
            rel_pos_zero_init (bool): 如果为True,则相对位置参数初始化为0(默认为True)
            window_size (int): 窗口大小(默认为0)
            global_attn_indexes (list): 使用全局注意力的块的索引(默认为())
        """
        super().__init__()

        self.img_size = img_size
        self.in_chans = in_chans
        self.args = args

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        ) # patch_embed.shape = [1, 64, 64, 3]

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList() # blocks.shape = [12, 64, 64, 768]
        for i in range(depth): # depth = 12
            block = Block(
                args= self.args,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential( # neck.shape = [1, 64, 64, 256], 脖子是一个卷积层, 用于降维, 降维后的维度为256
            nn.Conv2d( # 降维的卷积层
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans), # 降维后的维度进行归一化
            nn.Conv2d( # 降维的卷积层
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ), # 降维后的维度进行归一化
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class Block(nn.Module):
    """ 支持窗口注意力和残差传播块的Transformer块
    Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        args,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        scale: float = 0.5,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        参数:
            args (argparse.Namespace): 命令行参数。
            dim (int): 输入通道数。
            num_heads (int): 每个ViT块中的注意力头数。
            mlp_ratio (float): mlp隐藏维度与嵌入维度的比率。
            qkv_bias (bool): 如果为True，则在查询、键、值中添加可学习的偏置。
            norm_layer (nn.Module): 归一化层。
            act_layer (nn.Module): 激活层。
            use_rel_pos (bool): 如果为True，则在注意力图中添加相对位置嵌入。
            rel_pos_zero_init (bool): 如果为True，则将相对位置参数初始化为零。
            window_size (int): 窗口注意力块的窗口大小。如果等于0，则使用全局注意力。
            input_size (tuple(int, int) or None): 用于计算相对位置参数大小的输入分辨率。
        返回:
            torch.Tensor: 输出特征图。
        """
        super().__init__()
        self.args = args
        self.norm1 = norm_layer(dim) # 归一化层(输入通道数). (64, 64, 768)
        self.attn = Attention( # 多头注意力层: 1.计算Q,K,V 2.计算attention map 3.计算输出
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        ) # (64, 64, 768)
        self.MLP_Adapter = Adapter(dim, skip_connect=False)  # MLP-adapter, no skip connection
        self.Space_Adapter = Adapter(dim)  # with skip connection
        self.scale = scale
        self.Depth_Adapter = Adapter(dim, skip_connect=False)  # no skip connection

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        ## 3d branch
        if self.args.thd:
            hh, ww = x.shape[1], x.shape[2]
            depth = self.args.chunk
            xd = rearrange(x, '(b d) h w c -> (b h w) d c ', d=depth)
            # xd = rearrange(xd, '(b d) n c -> (b n) d c', d=self.in_chans)
            xd = self.norm1(xd)
            dh, _ = closest_numbers(depth)
            xd = rearrange(xd, 'bhw (dh dw) c -> bhw dh dw c', dh= dh)
            xd = self.Depth_Adapter(self.attn(xd))
            xd = rearrange(xd, '(b n) dh dw c ->(b dh dw) n c', n= hh * ww )

        x = self.norm1(x)
        x = self.attn(x)
        x = self.Space_Adapter(x)

        if self.args.thd:
            xd = rearrange(xd, 'b (hh ww) c -> b  hh ww c', hh= hh )
            x = x + xd
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn) + self.scale * self.MLP_Adapter(xn)
        return x


class Attention(nn.Module):
    """
    Multi-head Attention block with relative position embeddings.
    多头注意力块，带有相对位置嵌入。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        参数:
            dim (int): 输入通道数。
            num_heads (int): 每个ViT块中的注意力头数。
            qkv_bias (bool): 如果为True，则在查询、键、值中添加可学习的偏置。
            use_rel_pos (bool): 如果为True，则在注意力图中添加相对位置嵌入。
            rel_pos_zero_init (bool): 如果为True，则将相对位置参数初始化为零。
            input_size (tuple(int, int) or None): 用于计算相对位置参数大小的输入分辨率。
        返回:
            torch.Tensor: 输出特征图。
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5 # 缩放因子

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 3个线性层，分别计算Q,K,V
        self.proj = nn.Linear(dim, dim) # 投影层

        self.use_rel_pos = use_rel_pos # 是否使用相对位置编码
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 前传至 Attention
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1) # 重组: (B, H, W, C)
        x = self.proj(x) # 投影

        return x # 输出特征图


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    必要时分区为不重叠的带有填充的窗口。
    参数:
        x (torch.Tensor): 输入令牌带有[B, H, W, C]。
        window_size (int): 窗口大小。
    返回:
        windows: 分区后的窗口，带有[B * num_windows, window_size, window_size, C]。
        (Hp, Wp): 分区前的填充高度和宽度

    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    窗口取消分区为原始序列并删除填充。
    参数:
        windows (torch.Tensor): 带有[B * num_windows, window_size, window_size, C]的输入令牌。
        window_size (int): 窗口大小。
        pad_hw (Tuple[int, int]): 填充高度和宽度(Hp, Wp)。
        hw (Tuple[int, int]): 填充前的原始高度和宽度(H, W)。
    返回:
        x: 带有[B, H, W, C]的未分区的窗口。
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    根据查询和键大小的相对位置获取相对位置嵌入。
    参数:
        q_size (int): 查询q的大小。
        k_size (int): 键k的大小。
        rel_pos (Tensor): 相对位置嵌入(L, C)。
    返回:
        根据相对位置提取的位置嵌入。
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]




def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


def closest_numbers(target):
    """
    找到最接近的两个数字
    参数:
        target (int): 目标数字。
    返回:
        (a, b): 最接近的两个数字。
    """
    a = int(target ** 0.5)
    b = a + 1
    while True:
        if a * b == target:
            return (a, b)
        elif a * b < target:
            b += 1
        else:
            a -= 1


class PatchEmbed(nn.Module):
    """ 分片嵌入
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        参数:
            kernel_size (Tuple): 投影层的内核大小,默认为(16, 16)。
            stride (Tuple): 投影层的步长,默认为(16, 16)。
            padding (Tuple): 投影层的填充大小,默认为(0, 0)。
            in_chans (int): 输入图像通道数,默认为3。
            embed_dim (int): 分片嵌入维度,默认为768。
        返回:
            x (Tensor): 输入图像的分片嵌入。
        """
        super().__init__()

        self.proj = nn.Conv2d( # 2d卷积, 用于图像, 目的是将图像分片, 然后将每个分片映射到一个向量
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
