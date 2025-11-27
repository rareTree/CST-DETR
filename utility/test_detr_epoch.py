# 文件名: utility/test_detr_epoch.py
#
# (最终修正版 V3)
# 修正了因 per_file=True 导致的 B > 1 批处理索引BUG
#

import os
import numpy as np
import torch
import csv


# ---------------------------------------------------------------------------
# 主函数: test_detr_epoch
# ---------------------------------------------------------------------------
def test_detr_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    """
    评估函数 - 修正版

    此版本正确处理 per_file=True 的 DataGenerator，
    其中一个批次(Batch)中的 B 个序列全部来自同一个文件。
    """

    # --- 1. 设置 ---
    test_filelist = data_generator.get_filelist()  # 获取文件名列表
    nb_test_batches, test_loss = 0, 0.
    model.eval()  # 关键：将模型设置为评估模式

    # 禁用梯度计算
    with torch.no_grad():
        # --------------------------------------------------------------------
        # 修正点 1: 我们使用 enumerate 来获取 file_cnt (即 batch_idx)
        # --------------------------------------------------------------------
        # 循环的每一次迭代处理 *一个* 完整的文件
        for file_cnt, (data, target) in enumerate(data_generator.generate()):

            # --- 2. 计算验证损失 (val_loss) ---

            data = torch.tensor(data).float().to(device)
            target = torch.tensor(target).float().to(device)  # [B, T, N_gt, 4, K]
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            nb_test_batches += 1

            # --- 3. 解码模型输出 -> 写入 .csv 文件 ---

            pred_logits = output['pred_logits']  # [B, T, N, K+1]
            pred_doa = output['pred_doa']  # [B, T, N, 3]

            B, T, N, K_plus_1 = pred_logits.shape
            K = params['unique_classes']

            pred_scores = pred_logits.softmax(dim=-1)
            pred_classes = pred_scores.argmax(dim=-1)  # [B, T, N]

            # --------------------------------------------------------------------
            # 修正点 2: “缝合” (Stitch)
            # 我们必须将 B 个序列重新组合成一个文件
            # [B, T, N] -> [B*T, N] (e.g., [16, 64, 6] -> [1024, 6])
            # --------------------------------------------------------------------
            pred_classes_file = pred_classes.reshape(B * T, N).cpu().numpy()

            # [B, T, N, 3] -> [B*T, N, 3] (e.g., [16, 64, 6, 3] -> [1024, 6, 3])
            pred_doa_file = pred_doa.reshape(B * T, N, 3).cpu().numpy()

            # --------------------------------------------------------------------
            # 修正点 3: 获取正确的文件名
            # file_cnt 来自 enumerate，它总是正确的 (0, 1, 2, ...)
            # --------------------------------------------------------------------
            current_file_name = test_filelist[file_cnt]
            output_file = os.path.join(dcase_output_folder, current_file_name.replace('.npy', '.csv'))

            output_dict = {}

            # num_frames 现在是 B * T (e.g., 1024)
            num_frames = B * T

            # 遍历这个文件的 *所有* 帧 (B*T)
            for frame_cnt in range(num_frames):
                # 遍历这一帧的 N 个预测槽位
                for n_query in range(N):

                    # --- 正确的过滤逻辑 (和之前一样) ---
                    class_idx = pred_classes_file[frame_cnt, n_query]

                    if class_idx < K:  # 如果 "获胜者" 不是 "无事件"

                        doa_vector = pred_doa_file[frame_cnt, n_query]
                        x, y, z = doa_vector[0], doa_vector[1], doa_vector[2]

                        if frame_cnt not in output_dict:
                            output_dict[frame_cnt] = []
                        output_dict[frame_cnt].append([class_idx, x, y, z])

            # --------------------------------------------------------------------
            # 修正点 4: 写入 *一个* 文件
            # (我们不再有 b_idx 循环了，只在文件循环的末尾写入一次)
            # --------------------------------------------------------------------
            data_generator.write_output_format_file(
                output_file,
                output_dict,
            )

            if params['quick_test'] and nb_test_batches == 4:
                break

        test_loss /= nb_test_batches

    return test_loss
