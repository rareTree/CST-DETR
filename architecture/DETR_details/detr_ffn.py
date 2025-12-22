import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class DETR_SELD_Head(nn.Module):
    def __init__(self,  num_classes: int = 13, num_queries: int = 8, embed_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes  # K = 13
        self.num_queries = num_queries  # N = 6
        self.embed_dim = embed_dim  # C_embed = 64

        # ... (CST-former 和 DETR Decoder 主体) ...

        # 1. 分类头
        # 输出维度必须是 K + 1
        self.class_head = nn.Linear(embed_dim, self.num_classes + 1)  # -> [64, 14]

        # 2. DOA 头
        # 输出维度是 3 (x, y, z)
        self.doa_head = nn.Linear(embed_dim, 3)  # -> [64, 3]

    def forward(self, x):
        """
        输入：detr_output → [Q=250, B=32, C=64]
        输出：pred → [B=32, Q=250, 17]（17=激活+DOA+类别）
        """
        # 步骤1：调整维度顺序（Q,B,C）→（B,Q,C），批次维度前置
        # 此处torch.Size([1, 300, 32, 64])，只有第一个为1时这么用
        x = x.squeeze(dim=0).permute(1, 0, 2)  # [32, 250, 64]
        # [B, T*N, K+1]
        pred_logits = self.class_head(x)
        # [B, T*N, 3]
        pred_doa = self.doa_head(x)

        # --- Reshape 为 [B, T, N, ...] 格式 ---

        B, T = x.shape[0], 50  # 假设 T=50

        # [B, T*N, 14] -> [B, T, N, 14]
        out_logits = pred_logits.view(B, T, self.num_queries, self.num_classes + 1)

        # [B, T*N, 3] -> [B, T, N, 3]
        out_doa = pred_doa.view(B, T, self.num_queries, 3)

        # 打包成字典
        outputs = {
            'pred_logits': out_logits,
            'pred_doa': out_doa
        }

        return outputs
