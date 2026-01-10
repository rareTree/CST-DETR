import torch
import torch.nn as nn


class DETR_SELD_Head(nn.Module):
    def __init__(self, num_classes: int = 13, num_queries: int = 8, embed_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. 分类头 (输出 K 类)
        self.class_head = nn.Linear(embed_dim, self.num_classes)

        # 2. DOA 头 (输出 3 维坐标)
        self.doa_head = nn.Linear(embed_dim, 3)

    def forward(self, x):
        """
        输入：x → [Batch, Sequence_Length, Channel]
        输出：字典包含 logits 和 doa
        """
        # x 已经是 [B, T*N, C] 格式，直接通过 Linear 层即可

        # [B, T*N, K]
        pred_logits = self.class_head(x)

        # [B, T*N, 3]
        pred_doa = torch.tanh(self.doa_head(x))

        outputs = {
            'pred_logits': pred_logits,
            'pred_doa': pred_doa
        }
        return outputs