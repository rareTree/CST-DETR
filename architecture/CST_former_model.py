import numpy as np
import torch
import torch.nn as nn

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

        # 计算卷积后的频率维度：输入梅尔 bins数 / 频率池化尺寸的乘积（池化会降低频率维度）
        self.conv_block_freq_dim = int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.input_nb_ch = 7
        # 计算时间嵌入维度（注意力模块的输入维度）
        # 如果使用DCA模式，需要考虑麦克风通道数，否则仅基于卷积特征维度
        self.temp_embed_dim = self.conv_block_freq_dim * params['nb_cnn2d_filt'] * self.input_nb_ch if self.ch_attn_dca \
            else self.conv_block_freq_dim * params['nb_cnn2d_filt']

        ## Attention Layer===========================================================================================#
        if not self.cmt_block:
            # 不使用CMT模块时，使用CST_encoder（纯注意力机制）
            self.attention_stage = CST_encoder(self.temp_embed_dim, params)
        else:
            # 使用CMT模块时，使用卷积与Transformer混合结构
            self.attention_stage = CMT_block(params, self.temp_embed_dim)

        # 时间池化层：如果池化位置在'end'，根据频率池化配置选择池化核
        if self.t_pooling_loc == 'end':
            # 若频率池化未改变维度（f_pool_size全为1），使用(5,4)池化核；否则使用(5,1)
            if not params["f_pool_size"] == [1, 1, 1]:
                self.t_pooling = nn.MaxPool2d((5, 1))  # 时间维度池化（5倍下采样），频率维度不变
            else:
                self.t_pooling = nn.MaxPool2d((5, 4))  # 时间和频率维度同时池化

        ## Fully Connected Layer ======================================================================================#
        ## 全连接层：将注意力输出映射到最终结果（如声音事件类别和方位）

        self.positional_encoding = PositionalEncoding()
        self.query_generator = SELDQueryGenerator()
        # 1. 评分头 : 判断特征点是不是事件
        self.enc_score_head = nn.Linear(64, 1)
        # 2. 初始坐标预测头 (从 Encoder 特征直接猜 DOA)
        self.enc_doa_head = nn.Linear(64, 3)
        # 3. 坐标 -> 位置编码 的映射 MLP (Refinement 核心)
        # 将 3维坐标 映射回 64维 Pos Embedding
        self.ref_point_head = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        self.pos_trans_norm = nn.LayerNorm(128)  # 防止梯度爆炸
        self.decoder_layer = TransformerDecoderLayer(128, 4, 256)
        self.decoder_norm = nn.LayerNorm(128)
        self.decoder = TransformerDecoder(self.decoder_layer, 6, self.decoder_norm,
                                          return_intermediate=params['return_intermediate'])
        self.ffn = DETR_SELD_Head()

        # 初始化模型参数
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        参数初始化函数：为不同类型的层设置合适的初始化方式
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            # 线性层使用xavier_uniform初始化（适合激活函数为ReLU的场景）
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            # 卷积层使用kaiming_uniform初始化（适合ReLU激活）
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            # 层归一化的权重初始化为1，偏置为0
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, video=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        """
                前向传播函数
                input: x的形状为 (batch_size, mic_channels, time_steps, mel_bins) 
                       即（批量大小，麦克风通道数，时间步，梅尔频谱 bins）
                video: 预留参数（未使用，可能用于未来扩展多模态输入）
        """
        B, M, T, F = x.size()

        # 如果使用DCA通道注意力，调整输入形状：将麦克风通道合并到批量维度
        # 原形状 (B, M, T, F) → 重排为 (B*M, 1, T, F)，便于单独处理每个麦克风通道的特征
        if self.ch_attn_dca:
            x = rearrange(x, 'b m t f -> (b m) 1 t f', b=B, m=M, t=T, f=F).contiguous()
        if self.print_result:
            print(f"x.size:{x.shape}")

        # 经过卷积编码器提取初步特征
        # 输出形状：若使用DCA则为 [(B*M), C, T', F']，否则为 [B, C, T', F']（C为卷积输出通道数）
        if self.print_result:
            print("=================== encoder ====================")
        x = self.encoder(x)  # OUT : [(b m) c t f] if ch_attn_dca else [b c t f]
        if self.print_result:
            print(f"x after encoder to attention(x.size):{x.shape}")

            print("=================== attention ===================")
        # 经过注意力阶段（CST_encoder或CMT_block），融合通道-频谱-时间特征
        x = self.attention_stage(x)
        # 如果时间池化在末尾，应用池化层进一步降低维度
        if self.t_pooling_loc == 'end':
            x = self.t_pooling(x)
        x = rearrange(x, 'b t (f c) -> b c t f', f=16).contiguous()
        if self.print_result:
            print(f"x after attention(x.size):{x.shape}")



        if self.print_result:
            print("=================== DETR ===================")
        # 经过全连接层输出最终结果（DOA：方向角，包含声音事件类别和定位信息）

        # 加入位置编码
        pos = self.positional_encoding(x)
        pos = pos.permute(0, 2, 3, 1)
        memory = x.permute(0, 2, 3, 1)

        if self.print_result:
            print(f"memory after positional encoding(x.size):{memory.shape}")

        # 1. 给 Encoder 的所有特征点打分
        enc_logits = self.enc_score_head(memory)

        # 2. 逐帧筛选: 在 Freq 维度 (dim=2) 选出每帧最强的 6 个点
        num_q_per_frame = 6
        # topk_indices: [B, T, 6, 1]
        topk_scores, topk_indices = torch.topk(enc_logits, num_q_per_frame, dim=2)

        # 3. 提取特征 (tgt) 和位置 (pos)
        # 扩展索引维度以匹配特征维度: [B, T, 6, C]
        expand_indices = topk_indices.expand(-1, -1, -1, memory.shape[-1])

        # gather: [B, T, F, C] -> [B, T, 6, C]
        tgt_frame = torch.gather(memory, 2, expand_indices)
        pos_frame = torch.gather(pos, 2, expand_indices)

        # 4. 展平为 [B, L_query, C] 以输入 Transformer (L_query = T * 6 = 300)
        tgt = tgt_frame.flatten(1, 2)  # [B, 300, 64]
        query_pos = pos_frame.flatten(1, 2)  # [B, 300, 64]

        # 5. 预测初始参考点 (DOA)
        # 使用选出的特征直接预测初始坐标
        ref_points = self.enc_doa_head(tgt)  # [B, 300, 3]

        # 6. 转换维度顺序: [Sequence, Batch, Channel] (Transformer 标准)
        tgt = tgt.transpose(0, 1)  # [300, B, 64]
        query_pos = query_pos.transpose(0, 1)  # [300, B, 64]
        if self.print_result:
            print(f"tgt:{tgt.shape}, query_pos:{query_pos.shape}")

        # Memory 也要展平并转置: [B, T, F, C] -> [B, T*F, C] -> [T*F, B, C]
        memory = memory.flatten(1, 2).transpose(0, 1)  # [800, B, 64]
        pos = pos.flatten(1, 2).transpose(0, 1)
        if self.print_result:
            print(f"memory:{memory.shape}, pos_pos:{pos.shape}")

        # --- 迭代坐标细化 (Iterative Refinement) ---
        output = tgt
        current_pos = query_pos  # Layer 0 使用 Top-K 的原始 Pos
        current_ref_points = ref_points  # 用于记录参考点变化
        outputs_list = []
        for layer_idx, layer in enumerate(self.decoder.layers):
            # A. 运行当前层 Transformer
            # 注意：pos 是 encoder 的位置编码 (Keys)，query_pos 是 decoder 的位置编码 (Queries)
            output = layer(
                output,
                memory,
                pos=pos,
                query_pos=current_pos
            )

            # B. 归一化
            if self.decoder.norm is not None:
                output_norm = self.decoder.norm(output)
            else:
                output_norm = output

            # C. 预测这一层的输出 (分类 + DOA)
            # FFN 期望输入 [1, Q, B, C] (模拟 TransformerDecoder 的 unsqueeze 输出)
            layer_predictions = self.ffn(output_norm.unsqueeze(0))

            # 提取预测的 DOA: [B, T, N, 3] (ffn 内部做了 view) -> [B, T*N, 3] -> [B, Q, 3]
            # 注意: detr_ffn 的 forward 返回的 pred_doa 是 [B, T, 6, 3]
            # 我们需要把它展平为 [B, 300, 3] 才能进行下面的 MLP 映射
            pred_doa_layer = layer_predictions['pred_doa'].flatten(1, 2)

            # D. 更新下一层的 query_pos (如果是中间层)
            if layer_idx < len(self.decoder.layers) - 1:
                # 阻断梯度，防止位置生成的梯度回传
                new_ref_points = pred_doa_layer.detach()

                # 将坐标映射回位置编码
                new_pos_embed = self.ref_point_head(new_ref_points)  # [B, 300, 64]
                new_pos_embed = self.pos_trans_norm(new_pos_embed)

                # 转回 Seq First: [300, B, 64]
                current_pos = new_pos_embed.transpose(0, 1)

                # 更新当前参考点 (逻辑上)
                current_ref_points = new_ref_points

            # E. 收集输出
            outputs_list.append(layer_predictions)


        if self.print_result:
            print(f"outputs_list:{len(outputs_list)}")

        # 3. 组织输出格式
        # 取出最后一层作为“主输出”
        outputs = outputs_list[-1]

        # 如果有中间层，把它们打包进 'aux_outputs'
        if self.decoder.return_intermediate:
            # outputs_list[:-1] 是除了最后一层之外的所有层 (L1...L5)
            outputs['aux_outputs'] = outputs_list[:-1]

        if self.print_result:
            print(f"outputs:{outputs.keys()}")
        return outputs
