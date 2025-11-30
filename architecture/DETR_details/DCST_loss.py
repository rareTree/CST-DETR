# 文件名: architecture/DETR_details/DCST_loss.py
# (标准 GPU 版本)

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for Multi-class Classification
        Args:
            weight: (Tensor, optional) 每个类别的权重 (即 SetCriterion 中的 empty_weight)
            gamma: (float) 聚焦参数，通常设为 2.0。值越大，模型越关注难分样本。
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        """
        input_tensor: [N, C] (Logits, 未经过 softmax)
        target_tensor: [N] (类别索引)
        """
        # 1. 计算标准的 Log Softmax
        log_prob = F.log_softmax(input_tensor, dim=-1)

        # 2. 获取概率 p = exp(log_p)
        prob = torch.exp(log_prob)

        # 3. 获取对应目标类别的概率 pt
        # target_tensor.view(-1, 1) -> [N, 1]
        # gather 提取每个样本真实类别对应的预测概率
        pt = prob.gather(1, target_tensor.view(-1, 1))
        log_pt = log_prob.gather(1, target_tensor.view(-1, 1))

        # 4. 如果有类别权重 (如 eos_coef)，应用它
        # self.weight 是 [C] 维向量
        if self.weight is not None:
            if self.weight.device != input_tensor.device:
                self.weight = self.weight.to(input_tensor.device)

            # 获取每个样本对应的类别权重 alpha_t
            at = self.weight.gather(0, target_tensor.view(-1))
            log_pt = log_pt * at.view(-1, 1)

        # 5. 计算 Focal Loss 公式: - (1 - pt)^gamma * log(pt) * alpha
        # (1 - pt) 是调制因子：样本越容易分类 (pt大)，(1-pt)越小，损失权重越低
        loss = -1 * (1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 2.0, cost_doa: float = 5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_doa = cost_doa

    @torch.no_grad()
    def forward(self, outputs, targets):
        BT, N, _ = outputs["pred_logits"].shape
        indices = []
        for i in range(BT):
            out_prob = outputs["pred_logits"][i].softmax(-1)
            out_doa = outputs["pred_doa"][i]

            tgt_labels = targets[i]["labels"]
            tgt_doa = targets[i]["doa"]

            if tgt_labels.numel() == 0:
                indices.append((torch.tensor([], dtype=torch.long, device=out_prob.device),
                                torch.tensor([], dtype=torch.long, device=out_prob.device)))
                continue

            cost_class = -out_prob[:, tgt_labels]
            cost_doa = torch.cdist(out_doa, tgt_doa, p=1)

            C = self.cost_class * cost_class + self.cost_doa * cost_doa

            # scipy 必须在 CPU 上运行
            C = C.cpu()

            pred_idx, gt_idx = linear_sum_assignment(C)

            indices.append((torch.from_numpy(pred_idx).long().to(out_prob.device),
                            torch.from_numpy(gt_idx).long().to(out_prob.device)))

        return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.focal_loss = FocalLoss(weight=self.empty_weight, gamma=2.0, reduction='mean')

    def forward(self, outputs, targets_adpit):
        device = outputs['pred_logits'].device
        B, T, N, _ = outputs["pred_logits"].shape

        outputs_flat = {}
        outputs_flat['pred_logits'] = outputs['pred_logits'].reshape(-1, N, self.num_classes + 1)
        outputs_flat['pred_doa'] = outputs['pred_doa'].reshape(-1, N, 3)

        # 1. 转换 GT (GPU 上进行)
        targets_processed = self.convert_adpit_to_set(targets_adpit)

        # 2. 匈牙利匹配
        indices = self.matcher(outputs_flat, targets_processed)

        # 3. 计算损失
        src_idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.full(outputs_flat['pred_logits'].shape[:2],
                                      self.num_classes,
                                      dtype=torch.long,
                                      device=device)

        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets_processed, indices)], dim=0)
        target_classes_o[src_idx] = target_classes

        pred_logits_flat = outputs_flat['pred_logits'].flatten(0, 1)
        target_classes_flat = target_classes_o.flatten(0, 1)

        # Focal Loss 内部会自动使用我们传入的 empty_weight
        loss_class = self.focal_loss(pred_logits_flat, target_classes_flat)

        pred_doa_matched = outputs_flat['pred_doa'][src_idx]
        target_doa_matched = torch.cat([t['doa'][J] for t, (_, J) in zip(targets_processed, indices)], dim=0)

        if pred_doa_matched.shape[0] > 0:
            loss_doa = F.l1_loss(pred_doa_matched, target_doa_matched, reduction="mean")
        else:
            loss_doa = torch.tensor(0.0, device=device)

        losses = {'loss_class': loss_class, 'loss_doa': loss_doa}

        final_loss_dict = {}
        for k, v in losses.items():
            if k in self.losses and k in self.weight_dict and self.weight_dict[k] > 0:
                final_loss_dict[k] = v * self.weight_dict[k]

        return sum(final_loss_dict.values())

    def convert_adpit_to_set(self, targets_adpit: torch.Tensor):
        """
        标准 GPU 版本转换函数
        """
        B, T, N_gt, _, K = targets_adpit.shape
        device = targets_adpit.device

        # 加上 contiguous 以防万一，反正是在 GPU 上很快
        targets_permuted = targets_adpit.permute(0, 1, 4, 2, 3).contiguous()

        act_flags = targets_permuted[..., 0]
        doa_vectors = targets_permuted[..., 1:]

        act_flat = act_flags.reshape(-1, K, N_gt)
        doa_flat = doa_vectors.reshape(-1, K, N_gt, 3)

        targets_processed = []

        for i in range(B * T):
            frame_acts = act_flat[i]
            frame_doas = doa_flat[i]

            # GPU 上的 nonzero
            active_indices = (frame_acts > 0.5).nonzero(as_tuple=False)

            if active_indices.numel() == 0:
                targets_processed.append({
                    "labels": torch.tensor([], dtype=torch.long, device=device),
                    "doa": torch.tensor([], dtype=torch.float, device=device)
                })
            else:
                active_labels = active_indices[:, 0]
                active_slots = active_indices[:, 1]
                active_doas = frame_doas[active_labels, active_slots]

                targets_processed.append({
                    "labels": active_labels,
                    "doa": active_doas
                })

        return targets_processed

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_target_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx