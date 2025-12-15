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
        self.enc_score_head = nn.Linear(64, 1)
        self.num_queries = 300

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
        self.decoder_layer = TransformerDecoderLayer(64, 4, 256)
        self.decoder_norm = nn.LayerNorm(64)
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
        x = rearrange(x, 'b t (f c) -> b c t f', f=16).contiguous()
        if self.print_result:
            print(f"x after attention(x.size):{x.shape}")

        # 加入位置编码
        pos = self.positional_encoding(x)
        pos = rearrange(pos, 'b c t f -> b (t f) c').contiguous()
        memory = rearrange(x, 'b c t f -> b (t f) c').contiguous()
        if self.print_result:
            print(f"memory after positional encoding(x.size):{memory.shape}")

        # 1. 给 Encoder 的所有特征点打分
        enc_logits = self.enc_score_head(memory).squeeze(-1)

        # 2. 选出分数最高的 Top-K 个索引 (K = num_queries)
        K = self.num_queries
        topk_scores, topk_indices = torch.topk(enc_logits, K, dim=1)

        # 3. 提取 Top-K 的特征作为 Content Query (tgt)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).repeat(1, K)

        tgt = memory[batch_idx, topk_indices]
        query_pos = pos[batch_idx, topk_indices]
        # tgt, query_pos = self.query_generator(B)
        tgt = tgt.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        if self.print_result:
            print(f"tgt:{tgt.shape}, query_pos:{query_pos.shape}")

        memory = memory.permute(1, 0, 2)
        pos = pos.permute(1, 0, 2)
        if self.print_result:
            print(f"memory:{memory.shape}, pos_pos:{pos.shape}")

        hs = self.decoder(tgt, memory, query_pos=query_pos, pos=pos)
        if self.print_result:
            print(f"hs:{hs.shape}")
        # 如果时间池化在末尾，应用池化层进一步降低维度
        if self.t_pooling_loc == 'end':
            x = self.t_pooling(x)
        if self.print_result:
            print("=================== FC ===================")
        # 经过全连接层输出最终结果（DOA：方向角，包含声音事件类别和定位信息）
        outputs_list = []
        for i in range(hs.shape[0]):
            # hs[i] 是第 i 层的输出，形状 [Q, B, C]
            # ffn 期望输入 [1, Q, B, C] (因为它内部做了 squeeze(0))
            # 所以我们用 unsqueeze(0) 增加一个维度
            layer_out = self.ffn(hs[i].unsqueeze(0))
            outputs_list.append(layer_out)

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
