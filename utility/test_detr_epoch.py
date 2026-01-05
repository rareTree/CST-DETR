# utility/test_detr_epoch.py
import os
import torch
import numpy as np


def test_detr_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    test_filelist = data_generator.get_filelist()
    nb_test_batches, test_loss = 0, 0.
    model.eval()

    # ================= RT-DETR 核心配置 =================
    # 置信度阈值 (Confidence Threshold)
    # 高于此分数的预测框才会被保留。
    # 建议范围: 0.3 ~ 0.5。如果虚警(ER)高，调高它；如果召回(LR)低，调低它。
    confidence_threshold = 0.3
    # ===================================================

    with torch.no_grad():
        for file_cnt, (data, target) in enumerate(data_generator.generate()):
            data = torch.tensor(data).float().to(device)
            target = torch.tensor(target).float().to(device)

            output = model(data)

            # 计算 Loss (仅用于显示，不影响推理结果)
            # 注意: 你的 criterion 必须已经改为了兼容 VFL 的版本
            loss = criterion(output, target)
            test_loss += loss.item()
            nb_test_batches += 1

            pred_logits = output['pred_logits']  # [B, T, N, C]
            pred_doa = output['pred_doa']  # [B, T, N, 3]

            B, T, N, C = pred_logits.shape
            K = params['unique_classes']  # 13

            # ---------------- 核心修改: Sigmoid + 阈值过滤 ----------------

            # 1. 维度自适应处理
            # 情况 A: 你改了 detr_ffn.py，输出维度 C = 13 (K) -> 直接用
            # 情况 B: 你没改 detr_ffn.py，输出维度 C = 14 (K+1) -> 切掉最后一位背景
            if C == K + 1:
                valid_logits = pred_logits[..., :K]
            else:
                valid_logits = pred_logits

            # 2. Sigmoid 激活 (RT-DETR 机制)
            # 这里的每个分数都是独立的概率 (0~1)
            pred_scores = valid_logits.sigmoid()  # [B, T, N, K]

            # 3. 获取每个 Query 最可能的类别和分数
            # max_probs: [B, T, N] -> 每个 query 的最高分
            # pred_classes: [B, T, N] -> 对应的类别索引
            max_probs, pred_classes = pred_scores.max(dim=-1)

            # 4. 转为 CPU numpy 用于后续写入
            pred_classes_cpu = pred_classes.reshape(B * T, N).cpu().numpy()
            max_probs_cpu = max_probs.reshape(B * T, N).cpu().numpy()
            pred_doa_cpu = pred_doa.reshape(B * T, N, 3).cpu().numpy()

            # -------------------------------------------------------------

            current_file_name = test_filelist[file_cnt]
            output_file = os.path.join(dcase_output_folder, current_file_name.replace('.npy', '.csv'))
            output_dict = {}

            num_frames = B * T

            # 遍历每一帧和每一个 Query，根据阈值筛选结果
            for frame_cnt in range(num_frames):
                for n_query in range(N):
                    # 获取该 Query 的置信度
                    score = max_probs_cpu[frame_cnt, n_query]

                    # ★★★ 关键判别: 只有分数 > threshold 才认为是有效检测 ★★★
                    if score > confidence_threshold:
                        class_idx = pred_classes_cpu[frame_cnt, n_query]

                        # 再次防守性检查 (理论上 Sigmoid 后不会越界，但为了安全)
                        if class_idx < K:
                            doa_vector = pred_doa_cpu[frame_cnt, n_query]
                            x, y, z = doa_vector[0], doa_vector[1], doa_vector[2]

                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []

                            # 写入结果: [class, x, y, z]
                            output_dict[frame_cnt].append([class_idx, x, y, z])

            data_generator.write_output_format_file(output_file, output_dict)

            if params['quick_test'] and nb_test_batches == 4: break

        test_loss /= nb_test_batches
    return test_loss