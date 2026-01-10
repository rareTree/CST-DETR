# 文件名: architecture/DETR_details/DCST_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# -------------------------------------------------------------
# 1. Varifocal Loss (独立函数)
# -------------------------------------------------------------
def varifocal_loss(pred_logits, gt_score, alpha=0.75, gamma=2.0):
    """
    pred_logits: [N, K]
    gt_score:    [N, K] (0~1)
    """
    pred_score = pred_logits.sigmoid()
    weight = alpha * pred_score.pow(gamma) * (1 - gt_score) + gt_score
    loss = F.binary_cross_entropy_with_logits(pred_logits, gt_score, reduction='none')
    return (loss * weight).sum(dim=1).mean()

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 2.0, cost_doa: float = 5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_doa = cost_doa

    @torch.no_grad()
    def forward(self, outputs, targets):
        BT, N, K = outputs["pred_logits"].shape
        indices = []
        for i in range(BT):
            out_prob = outputs["pred_logits"][i].sigmoid()
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
    def __init__(self, num_classes, matcher, weight_dict, losses, eos_coef=0.1, use_vtm_loss=False, vtm_penalty=3.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        # 保存 VTM 配置
        self.use_vtm_loss = use_vtm_loss
        self.vtm_penalty = vtm_penalty

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def get_loss(self, pred_logits, pred_doa, targets_adpit):
        """
        内部辅助函数：计算单层的 Loss (分类 + 回归)
        """
        # 1. 整理输出形状
        # [B, T, N, K]
        B, T, N, K = pred_logits.shape
        outputs_flat = {}
        outputs_flat['pred_logits'] = pred_logits.reshape(-1, N, K)
        outputs_flat['pred_doa'] = pred_doa.reshape(-1, N, 3)

        # 2. 转换 GT (targets_adpit 已经在 GPU 上)
        # 注意：这里我们传入原始的 adpit targets，内部函数会处理
        # 为了避免重复计算 convert_adpit_to_set，我们可以在 forward 里算好传进来
        # 但为了接口简单，这里假设传入的是处理好的 targets_processed
        targets_processed = targets_adpit

        # 3. 匈牙利匹配
        indices = self.matcher(outputs_flat, targets_processed)

        # 4. 获取索引
        src_idx = self._get_src_permutation_idx(indices)

        # ================= Loss 计算核心 =================
        # 展平以便统一处理: [Total_Queries, K]
        src_logits = outputs_flat['pred_logits'].flatten(0, 1)  # [B*T*N, K]

        # 初始化 Target (全 0)
        target_score = torch.zeros_like(src_logits)

        # 获取匹配到的正样本信息
        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets_processed, indices)])
        flat_indices = src_idx[0] * N + src_idx[1]

        # 计算角度质量
        pred_doa_matched = outputs_flat['pred_doa'][src_idx]
        pred_doa_matched = F.normalize(pred_doa_matched, p=2, dim=-1)
        target_doa_matched = torch.cat([t['doa'][J] for t, (_, J) in zip(targets_processed, indices)])

        if len(target_classes) > 0:
            # Cosine Sim -> Angle Error
            cos_sim = (pred_doa_matched * target_doa_matched).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
            angle_error = torch.acos(cos_sim) * 180 / 3.14159265


            # 2. 梯形策略：18度以内满分，18度以外衰减
            safe_margin = 18.0  # 宽容区阈值
            max_error = 180.0  # 最大可能误差

            # 计算超出 18 度的部分
            excess_error = torch.clamp(angle_error - safe_margin, min=0.0)

            # 计算衰减区间的长度 (180 - 18 = 162)
            decay_range = max_error - safe_margin
            quality = torch.clamp(1.0 - excess_error / decay_range, min=0.0, max=1.0)

            # 填入 Target
            # src_idx[0] 是 batch_idx (在 B*T 维度), src_idx[1] 是 query_idx
            # 展平索引 = batch_idx * N + query_idx
            target_score[flat_indices, target_classes] = quality.type_as(src_logits)

        # VFL Loss
        loss_class = varifocal_loss(src_logits, target_score)
        # ===============================================

        if pred_doa_matched.shape[0] > 0:
            loss_doa = F.l1_loss(pred_doa_matched, target_doa_matched, reduction="mean")
        else:
            loss_doa = torch.tensor(0.0, device=pred_logits.device)

        return loss_class, loss_doa

    def forward(self, outputs, targets_adpit):
        """
        计算总 Loss (主输出 + 辅助输出)
        """
        # 1. 预处理 GT (只需做一次)
        targets_processed = self.convert_adpit_to_set(targets_adpit)

        # 2. 计算最后一层的主 Loss
        loss_class, loss_doa = self.get_loss(outputs['pred_logits'], outputs['pred_doa'], targets_processed)

        # 根据权重求和
        final_loss = loss_class * self.weight_dict['loss_class'] + \
                     loss_doa * self.weight_dict['loss_doa']

        # 3. ★★★ 计算辅助 Loss (Aux Loss) ★★★
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # 对每一层中间输出计算 loss
                loss_class_aux, loss_doa_aux = self.get_loss(
                    aux_outputs['pred_logits'],
                    aux_outputs['pred_doa'],
                    targets_processed
                )

                # 累加到总 Loss
                final_loss += loss_class_aux * self.weight_dict['loss_class'] + \
                              loss_doa_aux * self.weight_dict['loss_doa']

        # ==================== 新增：Encoder Loss ====================
        if 'enc_outputs' in outputs:
            enc_out = outputs['enc_outputs']

            # 使用同样的匹配和 Loss 逻辑
            # 注意：因为我们已经把 Encoder 输出 reshape 成了 [B, T, N, C]，
            # 所以可以直接复用 self.get_loss
            l_class_enc, l_doa_enc = self.get_loss(enc_out['pred_logits'], enc_out['pred_doa'],
                                                           targets_processed)

             # 将 Encoder Loss 加入总 Loss
             # 通常 Encoder 的权重可以和 Decoder 一样
            final_loss += l_class_enc * self.weight_dict['loss_class'] + \
                                  l_doa_enc * self.weight_dict['loss_doa']
        # ============================================================

        return final_loss

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