# 文件名: architecture/DETR_details/DCST_loss.py
# (标准 GPU 版本)

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


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
        # [B, T, N, K+1]
        B, T, N, _ = pred_logits.shape
        outputs_flat = {}
        outputs_flat['pred_logits'] = pred_logits.reshape(-1, N, self.num_classes + 1)
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

        # 5. 计算分类 Loss
        target_classes_o = torch.full(outputs_flat['pred_logits'].shape[:2],
                                      self.num_classes,
                                      dtype=torch.long,
                                      device=pred_logits.device)

        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets_processed, indices)], dim=0)
        target_classes_o[src_idx] = target_classes

        pred_logits_flat = outputs_flat['pred_logits'].flatten(0, 1)
        target_classes_flat = target_classes_o.flatten(0, 1)

        # A. 首先计算不缩减(reduction='none')的基础Loss
        # 这样我们能拿到每一个样本单独的Loss值
        # 注意：这里已经包含了 self.empty_weight (eos_coef) 的处理
        raw_ce_loss = F.cross_entropy(pred_logits_flat, target_classes_flat,
                                      weight=self.empty_weight, reduction='none')

        # B. 判断是否应用 VTM 惩罚
        if self.use_vtm_loss:
            # 1. 计算每个类别的预测概率
            probs = pred_logits_flat.softmax(dim=-1)

            # 2. 拿到每个样本对应“真实标签类别”的概率
            # gather: 从 [Total, Classes] 中取出 Target 对应的那一列概率
            target_probs = probs.gather(1, target_classes_flat.unsqueeze(1)).squeeze(1)

            # 3. 定义“犹豫的正样本”
            # 条件1: 是正样本 (Target 不等于 背景类索引 self.num_classes)
            is_positive = (target_classes_flat != self.num_classes)
            # 条件2: 信心不足 (Target 对应概率 < 0.5)
            is_hesitant = (target_probs < 0.5)

            # 4. 需要惩罚的索引
            needs_penalty = is_positive & is_hesitant

            # 5. 应用惩罚权重
            # 创建一个全 1 的权重向量
            vtm_weights = torch.ones_like(raw_ce_loss)
            # 把需要惩罚的地方设为 3.0 (vtm_penalty)
            vtm_weights[needs_penalty] = self.vtm_penalty

            # 6. 加权并求平均
            loss_class = (raw_ce_loss * vtm_weights).mean()

        else:
            # 如果不开启 VTM，直接求平均 (保持原逻辑)
            loss_class = raw_ce_loss.mean()

        # 6. 计算 DOA Loss
        pred_doa_matched = outputs_flat['pred_doa'][src_idx]
        target_doa_matched = torch.cat([t['doa'][J] for t, (_, J) in zip(targets_processed, indices)], dim=0)

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