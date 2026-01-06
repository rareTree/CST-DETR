import numpy as np
import torch
import torch.nn as nn
import math

from einops import rearrange
from torch.utils.data.datapipes.utils.decoder import Decoder

from .CST_details.encoder import Encoder
from .CST_details.CST_encoder import CST_encoder
from .CST_details.CMT_Block import CMT_block
from .CST_details.layers import FC_layer
from .DETR_details.position import PositionalEncoding, SELDQueryGenerator
from .DETR_details.transformer import TransformerDecoder, TransformerDecoderLayer
from .DETR_details.detr_ffn import DETR_SELD_Head


class CST_former(torch.nn.Module):
    """
    CST_former : Channel-Spectral-Temporal Transformer for SELD task
    """

    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.t_pooling_loc = params["t_pooling_loc"]
        self.ch_attn_dca = params['ChAtten_DCA']
        self.ch_attn_unfold = params['ChAtten_ULE']
        self.cmt_block = params['CMT_block']
        self.encoder = Encoder(in_feat_shape, params)

        self.print_result = params['print_result']

        # 计算卷积后的频率维度
        self.conv_block_freq_dim = int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.input_nb_ch = 7

        # 计算时间嵌入维度
        self.temp_embed_dim = self.conv_block_freq_dim * params['nb_cnn2d_filt'] * self.input_nb_ch if self.ch_attn_dca \
            else self.conv_block_freq_dim * params['nb_cnn2d_filt']

        ## Attention Layer ===========================================================================================#
        if not self.cmt_block:
            self.attention_stage = CST_encoder(self.temp_embed_dim, params)
        else:
            self.attention_stage = CMT_block(params, self.temp_embed_dim)

        # 时间池化层
        if self.t_pooling_loc == 'end':
            if not params["f_pool_size"] == [1, 1, 1]:
                self.t_pooling = nn.MaxPool2d((5, 1))
            else:
                self.t_pooling = nn.MaxPool2d((5, 4))

        ## Fully Connected Layer (DETR) ==============================================================================#

        # [修改 1] 动态获取特征维度 (对于 Large 模型，这里应该是 128)
        # 如果 params 中没有 nb_cnn2d_filt，请确保它存在，或者用默认值 64
        embed_dim = params.get('nb_cnn2d_filt', 64)

        self.positional_encoding = PositionalEncoding()  # 确保 PosEnc 也用这个维度
        self.query_generator = SELDQueryGenerator(d_model=embed_dim)  # 确保 QueryGen 也用这个维度

        # [修改 2] 替换所有硬编码的 64 为 embed_dim
        # 1. 评分头
        self.enc_score_head = nn.Linear(embed_dim, self.nb_classes)
        # 2. 初始坐标预测头
        self.enc_doa_head = nn.Linear(embed_dim, 3)
        # 3. 坐标 -> 位置编码 的映射 MLP
        self.ref_point_head = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.pos_trans_norm = nn.LayerNorm(embed_dim)

        # Transformer Decoder 设置
        # 注意: nhead=4, 128/4=32 (整除), 64/4=16 (整除), 所以 nhead=4 是安全的
        self.decoder_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=256)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        self.decoder = TransformerDecoder(self.decoder_layer, 6, self.decoder_norm,
                                          return_intermediate=params['return_intermediate'])

        # [注意] 这里可能还需要检查 DETR_SELD_Head 内部是否写死了 64
        # 如果训练再次报错 shape mismatch 在 detr_ffn，请去修改 detr_ffn.py
        self.ffn = DETR_SELD_Head(embed_dim=embed_dim)  # 假设 DETR_SELD_Head 接受 input_dim 参数，如果不接受可能需要改那个文件

        # 初始化模型参数
        self.apply(self._init_weights)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # 1. 修改 Encoder 的分类头
        if hasattr(self, 'enc_score_head'):
            torch.nn.init.constant_(self.enc_score_head.bias, bias_value)

        # 2. 修改 Decoder 的分类头 (如果你能访问到的话)
        # 通常在 self.ffn 内部。如果没有报错，可以尝试加上这句：
        if hasattr(self.ffn, 'class_head'):
            torch.nn.init.constant_(self.ffn.class_head.bias, bias_value)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, video=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        B, M, T, F = x.size()

        if self.ch_attn_dca:
            x = rearrange(x, 'b m t f -> (b m) 1 t f', b=B, m=M, t=T, f=F).contiguous()
        if self.print_result:
            print(f"x.size:{x.shape}")

        if self.print_result:
            print("=================== encoder ====================")
        x = self.encoder(x)
        if self.print_result:
            print(f"x after encoder to attention(x.size):{x.shape}")

            print("=================== attention ===================")
        x = self.attention_stage(x)
        # 时间池化
        if self.t_pooling_loc == 'end':
            x = self.t_pooling(x)

        # [修改 3] 动态 rearrange，不再硬编码 f=16
        # 使用 self.conv_block_freq_dim (这取决于你的 Pooling 设置)
        # 如果你现在的配置算出来正好是 16，那就没问题。如果是 8，这行代码会自动适配。
        x = rearrange(x, 'b t (f c) -> b c t f', f=self.conv_block_freq_dim).contiguous()

        if self.print_result:
            print(f"x after attention(x.size):{x.shape}")

        if self.print_result:
            print("=================== DETR ===================")

        # 加入位置编码
        pos = self.positional_encoding(x)
        pos = pos.permute(0, 2, 3, 1)
        memory = x.permute(0, 2, 3, 1)

        if self.print_result:
            print(f"memory after positional encoding(x.size):{memory.shape}")

        # 1. 给 Encoder 的所有特征点打分
        # memory shape: [B, T, F, 128] -> enc_score_head -> [B, T, F, 13]
        enc_logits = self.enc_score_head(memory)

        # 2. 预测所有点的坐标 (Dense Prediction)
        # memory: [B, T, F, C] -> enc_doa: [B, T, F, 3]
        # 注意：这里我们对所有点都预测了 DOA，是为了后面方便取 Top-K
        enc_doa = self.enc_doa_head(memory)

        # 3. 逐帧筛选 (Top-K)
        # 现在的 enc_logits 是多分类的，我们需要找到“最大置信度”
        # 逻辑：先对 13 个类做 Max，得到“这是某个声音的概率”，再选 Top-K

        # [B, T, F, 13] -> [B, T, F] (取每个点最大的类分数)
        # 注意：这里我们用 Sigmoid 后的分数来排序，还是用 Logits 排序？
        # 为了数值稳定性，直接用 Logits 的 Max 值排序即可（单调性一致）
        enc_prob_max = enc_logits.max(dim=-1).values

        num_q_per_frame = 8
        topk_scores, topk_indices = torch.topk(enc_prob_max, num_q_per_frame, dim=2)

        # 4. 提取特征 (tgt) 和位置 (pos)
        # 需要扩展索引维度以匹配特征维度
        # topk_indices: [B, T, N] -> [B, T, N, C]
        expand_indices_C = topk_indices.unsqueeze(-1).expand(-1, -1, -1, memory.shape[-1])
        # topk_indices: [B, T, N] -> [B, T, N, 3]
        expand_indices_3 = topk_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        # topk_indices: [B, T, N] -> [B, T, N, 13]
        expand_indices_K = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.nb_classes)

        tgt_frame = torch.gather(memory, 2, expand_indices_C)

        # Gather Encoder 的预测结果 (用于 Loss 监督)
        # 选出来的这 8 个点的分类 Logits
        enc_topk_logits = torch.gather(enc_logits, 2, expand_indices_K)  # [B, T, N, 13]
        # 选出来的这 8 个点的 DOA
        enc_topk_doa = torch.gather(enc_doa, 2, expand_indices_3)  # [B, T, N, 3]

        # 5. 准备 Decoder 输入
        # 这里有一个重要改动：RT-DETR 通常直接用 Encoder 预测的 DOA 作为初始参考点
        # 而不再需要用 tgt 再过一次 enc_doa_head (虽然你之前的代码是这么做的)
        # 为了保持你代码的连贯性，我们这里可以让 ref_points 直接等于我们 gather 出来的 enc_topk_doa

        # 4. 展平
        tgt = tgt_frame.flatten(1, 2)
        tgt = tgt.transpose(0, 1)

        # 5. 预测初始参考点 (DOA)
        # 使用 Encoder 预测好的 DOA 作为初始参考点
        ref_points = enc_topk_doa.flatten(1, 2)  # [B, T*N, 3]

        enc_pos_embed = self.ref_point_head(ref_points)
        enc_pos_embed = self.pos_trans_norm(enc_pos_embed)
        current_pos = enc_pos_embed.transpose(0, 1)


        # Memory 展平并转置
        memory = memory.flatten(1, 2).transpose(0, 1)
        pos = pos.flatten(1, 2).transpose(0, 1)
        if self.print_result:
            print(f"memory:{memory.shape}, pos_pos:{pos.shape}")

        # --- 迭代坐标细化 ---
        output = tgt
        current_ref_points = ref_points
        outputs_list = []
        for layer_idx, layer in enumerate(self.decoder.layers):
            output = layer(
                output,
                memory,
                pos=pos,
                query_pos=current_pos
            )

            if self.decoder.norm is not None:
                output_norm = self.decoder.norm(output)
            else:
                output_norm = output

            # FFN
            layer_predictions = self.ffn(output_norm.permute(1, 0, 2))
            pred_doa_layer = layer_predictions['pred_doa']  # [B, T*N, 3]

            if layer_idx < len(self.decoder.layers) - 1:
                new_ref_points = pred_doa_layer.detach()
                new_pos_embed = self.ref_point_head(new_ref_points)
                new_pos_embed = self.pos_trans_norm(new_pos_embed)
                current_pos = new_pos_embed.transpose(0, 1)

                current_ref_points = new_ref_points

            B_new = B
            N_new = num_q_per_frame
            # 动态计算 T (因为可能经过了 Pooling)
            T_new = layer_predictions['pred_logits'].shape[1] // N_new

            layer_predictions['pred_logits'] = layer_predictions['pred_logits'].reshape(B_new, T_new, N_new, -1)
            layer_predictions['pred_doa'] = layer_predictions['pred_doa'].reshape(B_new, T_new, N_new, -1)

            outputs_list.append(layer_predictions)



        if self.print_result:
            print(f"outputs_list:{len(outputs_list)}")

        outputs = outputs_list[-1]

        if self.decoder.return_intermediate:
            outputs['aux_outputs'] = outputs_list[:-1]

        outputs['enc_outputs'] = {
            'pred_logits': enc_topk_logits,
            'pred_doa': enc_topk_doa
        }

        return outputs