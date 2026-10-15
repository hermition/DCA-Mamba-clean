# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .layers import (ConvLayer, DSConv, MBConv, FusedMBConv, IdentityLayer,
                     ResidualBlock, OpSequential, DAGBlock, UpSampleLayer)
from .mamba import VSSBlock

class EfficientViTBackboneMamba(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
        input_size=224,
        # ss2d_type='MHSS2DCas',
        # k=4,
        # mix_type='mlp',
        # f_fusion=None
    ) -> None:
        super().__init__()
        local_input_size = input_size
        # load data: 2794, 150, 1600
        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        local_input_size = local_input_size//2

        CBAM_ker_size_list = [7, 7]

        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        spilt_index = 3

        for index, (w, d) in enumerate(zip(width_list[1:spilt_index], depth_list[1:spilt_index])):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                in_channels = w
                stage.append(block)
                if stride==2:
                    local_input_size = local_input_size//2

            # stage.append(Rearrange('b c h w -> b w h c'))
            # if index==1:
            #     for _ in range(2):
            #         stage.append(VSSBlock(hidden_dim=in_channels, ss2d_type=ss2d_type))
            # stage.append(Rearrange('b w h c -> b c h w'))

            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for index, (w, d) in enumerate(zip(width_list[spilt_index:], depth_list[spilt_index:])):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )

            stage.append(ResidualBlock(block, None))
            in_channels = w
            local_input_size = local_input_size//2

            # for _ in range(mamaba_depth[index+3]):
            #     stage.append(ConvDeformMamba(in_channels,
            #                              in_channels,
            #                              ss2d_type=ss2d_type))

            # for _ in range(d):
            #     stage.append(EfficientViTBlockMamba(
            #             in_channels=in_channels,
            #             ss2d_type=ss2d_type))

            # mamba attention
            stage.append(Rearrange('b c h w -> b w h c'))


            for _ in range(d):
                    # mamba_atten = VSSBlock(hidden_dim=in_channels,
                    #                       ss2d_type=ss2d_type,
                    #                       k=k,
                    #                       mix_type=mix_type,
                    #                       f_fusion=f_fusion,
                    #                       size=local_input_size)
                    # stage.append(ResidualBlock(mamba_atten, IdentityLayer()))
                    # print(f's{index}_l{_}')
                    stage.append(
                        VSSBlock(hidden_dim=in_channels,
                                 name=f's{index}_l{_}',
                                 CBAM_ker_size = CBAM_ker_size_list[index],
                                ))

            stage.append(Rearrange('b w h c -> b c h w'))

            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor, meta=None) -> dict[str, torch.Tensor]:
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict["stage%d" % stage_id] = x = stage(x, meta)

        output_dict["stage_final"] = x
        return output_dict

class SegHead(DAGBlock):
    def __init__(
        self,
        fid_list: list[str],
        in_channel_list: list[int],
        stride_list: list[int],
        head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        middle_op: str,
        final_expand: float or None,
        n_classes: int,
        dropout=0,
        norm="bn2d",
        act_func="hswish",
    ):
        inputs = {}
        for fid, in_channel, stride in zip(fid_list, in_channel_list, stride_list):
            factor = stride // head_stride
            if factor == 1:
                inputs[fid] = ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None)
            else:
                inputs[fid] = OpSequential(
                    [
                        ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                        UpSampleLayer(factor=factor),
                    ]
                )

        middle = []
        for _ in range(head_depth):
            block = FusedMBConv(
                head_width,
                head_width,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=(act_func, None),
            )
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "segout": OpSequential(
                [
                    None
                    if final_expand is None
                    else ConvLayer(head_width, head_width * final_expand, 1, norm=norm, act_func=act_func),
                    ConvLayer(
                        head_width * (final_expand or 1),
                        n_classes,
                        1,
                        use_bias=True,
                        dropout=dropout,
                        norm=None,
                        act_func=None,
                    ),
                    # nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                    UpSampleLayer(factor=4),
                ]
            )
        }

        super(SegHead, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)

class EfficientViTSeg(nn.Module):
    def __init__(self, backbone: EfficientViTBackboneMamba, head: SegHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        # self.loadAndFreeze(False)

    def forward(self, x: torch.Tensor, meta=None) -> torch.Tensor:
        feed_dict = self.backbone(x)
        feed_dict = self.head(feed_dict)
        return feed_dict["segout"]



def build_model():
    """构造指定 DCA ring 256 实验；返回未加载权重的单类别 logits 模型。"""
    backbone = EfficientViTBackboneMamba(
        width_list=[32, 64, 128, 256, 512], depth_list=[1, 2, 3, 3, 6], input_size=256)
    head = SegHead(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128], stride_list=[64, 32, 16],
        head_stride=8, head_width=128, head_depth=3, expand_ratio=4,
        middle_op="fmbconv", final_expand=None, n_classes=1, act_func="gelu")
    return EfficientViTSeg(backbone, head)
