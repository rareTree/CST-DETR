import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from architecture.CST_details.CST_encoder import CST_attention
from .layers import LocalPerceptionUint, InvertedResidualFeedForward, Temp_attention, Spec_attention

class CMT_Layers(torch.nn.Module):
    def __init__(self, params, temp_embed_dim, ffn_ratio=4., drop_path_rate=0.):
        super().__init__()
        self.cmt_split = params['CMT_split']
        self.ch_attn_dca = params['ChAtten_DCA']
        self.ch_attn_ule = params['ChAtten_ULE']
        self.temp_embed_dim = temp_embed_dim
        self.ffn_ratio = ffn_ratio
        self.dim = params['nb_cnn2d_filt']

        self.norm1 = nn.LayerNorm(self.dim)
        self.LPU = LocalPerceptionUint(self.dim)
        self.IRFFN = InvertedResidualFeedForward(self.dim, self.ffn_ratio)

        self.print_result = params['print_result']

        if not self.cmt_split:
            self.cst_attention = CST_attention(temp_embed_dim=self.temp_embed_dim,params=params)
        elif self.cmt_split:
            self.spectral_atten = Spec_attention(temp_embed_dim=self.temp_embed_dim, params=params)
            self.temporal_atten = Temp_attention(temp_embed_dim=self.temp_embed_dim, params=params)
            self.norm2 = nn.LayerNorm(self.dim)
            self.LPU2 = LocalPerceptionUint(self.dim)
            self.IRFFN2 = InvertedResidualFeedForward(self.dim, self.ffn_ratio)

        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.print_result:
            print("This is CMT_Layers")
        if not self.cmt_split:
            # 不使用DCA注意力（使用ULE或DST注意力）
            if not self.ch_attn_dca: # ch_attn (ULE) or dst_attn
                if self.print_result:
                    print(f"It is into LPU(x),x.shape:{x.shape}")
                lpu = self.LPU(x)
                if self.print_result:
                    print("此处经过残差连接 x = x + lpu")
                x = x + lpu
                if self.print_result:
                    print(f"现在 x.shape is {x.shape}")

                B, C, T, F = x.size()
                x = rearrange(x, 'b c t f -> b t (f c)')
                if self.print_result:
                    print(f"x 经过 x = rearrange(x, 'b c t f -> b t (f c)') 后，x.shape is {x.shape}")
                if not self.ch_attn_dca: # channel attention with unfolding
                    if self.print_result:
                        print("此处进入cst_attention")
                    x = self.cst_attention(x,7,C, T, F)
                else:   # dst attention
                    x = self.cst_attention(x,C,T,F)

                if self.print_result:
                    print("回到CMT_Layers")
                    print(f"现在x的维度为：{x.shape}")
                x_2 = rearrange(x, 'b t (f c) -> b (t f) c', f=F).contiguous()
                if self.print_result:
                    print(f"经过 x_2 = rearrange(x, 'b t (f c) -> b (t f) c', f=F).contiguous() 后，x_2.shape is {x_2.shape}")
                x_res = rearrange(x, 'b t (f c) -> b c t f', f=F).contiguous()
                if self.print_result:
                    print(f"经过 x_res = rearrange(x, 'b t (f c) -> b c t f', f=F).contiguous() 后，x_res.shape is {x_res.shape}")
                norm1 = self.norm1(x_2)
                if self.print_result:
                    print("此处为 norm1 = self.norm1(x_2)")
                norm1 = rearrange(norm1, 'b (t f) c -> b c t f', f=F).contiguous()
                if self.print_result:
                    print(f"经过 norm1 = rearrange(norm1, 'b (t f) c -> b c t f', f=F).contiguous() 后，norm1.shape is {norm1.shape}")
                    print("这里进入IRFFN")
                ffn = self.IRFFN(norm1)
                if self.print_result:
                    print(f"经过ffn = self.IRFFN(norm1)，x.shape is {ffn.shape}")
                x = x_res + self.drop_path(ffn)
                if self.print_result:
                    print(f"残差连接，x = x_res + self.drop_path(ffn)，最后的维度为：{x.shape}")

            if self.ch_attn_dca: # ch_attn (DCA)
                B, C, T, F = x.size()
                M = 7

                lpu = self.LPU(x)
                x = x + lpu

                x = rearrange(x, '(b m) c t f -> b t (m f c)', m=M).contiguous()
                x = self.cst_attention(x, M, C, T, F)

                x_2 = rearrange(x, 'b t (m f c) -> b (t m f) c', m=M, f=F).contiguous()
                x_res = rearrange(x, 'b t (m f c) -> (b m) c t f', m=M, f=F).contiguous()
                norm1 = self.norm1(x_2)
                norm1 = rearrange(norm1, 'b (t m f) c -> (b m) c t f', f=F, c=C, t=T).contiguous()
                ffn = self.IRFFN(norm1)
                x = x_res + self.drop_path(ffn)

        else: # CMT Split
            if not self.ch_attn_dca and not self.ch_attn_ule:
                # Spectral Conformer
                lpu = self.LPU(x)
                x = x + lpu

                B, C, T, F = x.size()
                x = rearrange(x, 'b c t f -> b t (f c)').contiguous()
                x = self.spectral_atten(x, C, T, F)

                x_s = rearrange(x, 'b t (f c) -> b (t f) c', f=F).contiguous()
                x_res = rearrange(x, 'b t (f c) -> b c t f', f=F).contiguous()
                norm1 = self.norm1(x_s)
                norm1 = rearrange(norm1, 'b (t f) c -> b c t f', f=F).contiguous()
                ffn_s = self.IRFFN(norm1)
                xs = x_res + self.drop_path(ffn_s)

                # Temporal Conformer
                lpu2 = self.LPU2(xs)
                xs = xs + lpu2

                B, C, T, F = xs.size()
                x2 = rearrange(xs, 'b c t f -> b t (f c)').contiguous()
                x2 = self.temporal_atten(x2, C, T, F)

                x_t = rearrange(x2, 'b t (f c) -> b (t f) c', f=F).contiguous()
                x_res_t = rearrange(x2, 'b t (f c) -> b c t f', f=F).contiguous()
                norm2 = self.norm2(x_t)
                norm2 = rearrange(norm2, 'b (t f) c -> b c t f', f=F).contiguous()
                ffn_t = self.IRFFN2(norm2)
                x = x_res_t + self.drop_path(ffn_t)
            else:
                print("CST attention with split cmt block is not implemented yet.")
                raise()
        return x

class CMT_block(torch.nn.Module):
    def __init__(self, params, temp_embed_dim, ffn_ratio=4., drop_path_rate=0.1):
        super().__init__()
        self.temp_embed_dim = temp_embed_dim
        self.num_layers = params['nb_self_attn_layers']  # 2
        self.ch_atten_dca = params['ChAtten_DCA']
        self.ffn_ratio = ffn_ratio
        self.nb_ch = 7
        self.print_result=params['print_result']

        self.block_list = nn.ModuleList([CMT_Layers(
            params=params,
            temp_embed_dim=self.temp_embed_dim,
            ffn_ratio=self.ffn_ratio,
            drop_path_rate=drop_path_rate
        ) for i in range(self.num_layers)]
        )

    def forward(self, x):
        if self.print_result:
            print(f"This is CMT_block, the input size is {x.shape}")
        B, C, T, F = x.size()
        M = self.nb_ch

        for block in self.block_list:
            if self.print_result:
                print(f"This is block_list, it has 2 CMT_Layers")
            x = block(x)
        if self.print_result:
            print("block_list结束")

        if self.ch_atten_dca: # CST (DCA)
            B = B // M
            x = rearrange(x, '(b m) c t f -> b t (m f c)', b=B,m=M).contiguous()
        else: # CST (ULE) & DST
            if self.print_result:
                print(f"现在x.shape is {x.shape}")
            x = rearrange(x, 'b c t f -> b t (f c)', c=C, t=T, f=F).contiguous()
            if self.print_result:
                print(f"经过 x = rearrange(x, 'b c t f -> b t (f c)', c=C, t=T, f=F).contiguous() 后，x.shape is {x.shape}")

        return x
