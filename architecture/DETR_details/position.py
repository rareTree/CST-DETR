import torch
import torch.nn as nn
import math  # <--- 必须导入 math 库
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, max_T=150, max_F=32):
        """
        适配CST-former的双维度（时间+频率）位置编码
        Args:
            max_T: 最大时间帧长度（需 ≥ 实际T，如50）
            max_F: 最大频率bin数（需 ≥ 实际F，如16）
        """
        super().__init__()
        self.max_T = max_T  # 覆盖可能的时间帧长度
        self.max_F = max_F  # 覆盖可能的频率bin数

    def _generate_1d_pe(self, max_len, d_model, device):
        """生成1D正弦余弦位置编码（时间或频率）"""
        position = torch.arange(max_len, device=device).unsqueeze(1)  # [max_len, 1]

        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model, device=device)  # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引用余弦
        return pe  # [max_len, d_model]

    def forward(self, x):
        """
        输入：编码器输出特征 x，形状 [B, C, T, F]
        输出：叠加位置编码后的特征，形状 [B, C, T, F]
        """
        B, C, T, F = x.shape  # 动态获取维度：C=64, T=50, F=16（示例）
        device = x.device

        # 时间和频率编码各分配一半通道（C=64 → 各32通道）
        d_time = C // 2  # 32
        d_freq = C - d_time  # 32（确保总和为C）

        # 1. 生成时间位置编码：[T, d_time] → 扩展为 [T, 1, d_time] → 广播到 [T, F, d_time]
        pe_time = self._generate_1d_pe(self.max_T, d_time, device)[:T]  # [T, d_time]
        pe_time = pe_time.unsqueeze(1)  # [T, 1, d_time]
        pe_time = pe_time.expand(-1, F, -1)  # [T, F, d_time]（沿频率维度广播）

        # 2. 生成频率位置编码：[F, d_freq] → 扩展为 [1, F, d_freq] → 广播到 [T, F, d_freq]
        pe_freq = self._generate_1d_pe(self.max_F, d_freq, device)[:F]  # [F, d_freq]
        pe_freq = pe_freq.unsqueeze(0)  # [1, F, d_freq]
        pe_freq = pe_freq.expand(T, -1, -1)  # [T, F, d_freq]（沿时间维度广播）

        # 3. 合并时间和频率编码 → 调整维度为 [1, C, T, F]
        pe = torch.cat([pe_time, pe_freq], dim=-1)  # [T, F, C]（C = d_time + d_freq）
        pe = rearrange(pe, 't f c -> 1 c t f')  # 转换为 [1, C, T, F]，适配输入维度
        pe = pe.repeat(B, 1, 1, 1)

        # 4. 叠加位置编码（广播匹配批量维度B）
        return pe


class SELDQueryGenerator(nn.Module):
    def __init__(self, num_frames=50, num_queries_per_frame=8, d_model=128, max_T=50, max_F=16):
        super().__init__()
        self.num_frames = num_frames
        self.num_queries_per_frame = num_queries_per_frame
        self.total_queries = num_frames * num_queries_per_frame  # 50×5=250
        self.d_model = d_model
        self.d_time = d_model // 2  # 时间编码32维
        self.d_freq = d_model - self.d_time  # 频率编码32维

        # 1. 可学习的基础query嵌入
        self.base_queries = nn.Embedding(self.total_queries, d_model)  # [250, 64]

        # 2. 帧级时间编码（直接生成并注册，不手动创建属性）
        frame_time_pe = self._generate_1d_pe(max_T, self.d_time)  # [50, 32]
        self.register_buffer('frame_time_pe', frame_time_pe)  # 自动创建self.frame_time_pe

        # 3. 频率范围编码（同理，直接生成并注册）
        query_freq_pe = self._generate_1d_pe(max_F, self.d_freq)  # [16, 32]
        self.register_buffer('query_freq_pe', query_freq_pe)  # 自动创建self.query_freq_pe

    def _generate_1d_pe(self, max_len, d_embed):
        """生成正弦余弦编码"""
        # 注意：__init__ 中通常在 CPU 上运行，但也建议改用 math 以保持一致性
        position = torch.arange(max_len).unsqueeze(1)

        # ★★★ 修改点 2：使用 math.log ★★★
        div_term = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))

        pe = torch.zeros(max_len, d_embed)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, batch_size):
        # 1. 基础query：[250, 64] → [B, 250, 64]
        tgt_embed = self.base_queries.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # 注意：这里你使用了 tgt = zeros，这意味着 base_queries 的权重实际上被忽略了。
        # 如果你的意图是 query = Learned_Embedding + Positional_Encoding，
        # 那么下面应该是 queries = tgt_embed + combined_pe
        # 但为了保持和你原逻辑一致（只修报错），我保留了原样。
        tgt = torch.zeros_like(tgt_embed)

        # 2. 叠加帧级时间编码
        frame_time_pe_expand = self.frame_time_pe.unsqueeze(1).repeat(1, self.num_queries_per_frame, 1)  # [50,5,32]
        frame_time_pe_flat = frame_time_pe_expand.flatten(0, 1)  # [250, 32]

        # 3. 叠加频率范围编码
        freq_pe_per_query = self.query_freq_pe[:self.num_queries_per_frame]  # [5, 32]
        query_freq_pe_expand = freq_pe_per_query.unsqueeze(0).repeat(self.num_frames, 1, 1)  # [50,5,32]
        query_freq_pe_flat = query_freq_pe_expand.flatten(0, 1)  # [250, 32]

        # 4. 拼接编码并叠加到query
        combined_pe = torch.cat([frame_time_pe_flat, query_freq_pe_flat], dim=-1).unsqueeze(0)  # [1,250,64]

        # 再次提醒：这里 queries = 0 + PE，base_queries 没有被用到。
        # 如果这是故意的（anchor DETR风格），则无视。否则请检查这里是否应为 tgt_embed
        queries = tgt + combined_pe  # [B, 250, 64]

        # 返回 tgt (全0) 和 queries (全PE)
        return tgt, queries