import os
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def test_detr_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    """
    [修正版] 支持 IO + Polyphony + 数据缝合
    """
    test_filelist = data_generator.get_filelist()
    nb_test_batches, test_loss = 0, 0.
    model.eval()

    # --- IO 参数配置 ---
    CHUNK_LEN_SEC = 5.0  # 必须与训练时的 feature_sequence_length 一致
    HOP_LEN_SEC = 1.0  # 想要 1秒 滑动

    # 假设 1秒 = 50 帧 (取决于你的特征提取参数，通常是 100帧 或 50帧)
    # 自动推断帧率: 从 params['feature_sequence_length'] (500) 和 5秒 推算
    # 500帧 / 5秒 = 100帧/秒
    FEAT_SEQ_LEN = params['feature_sequence_length']  # 比如 500
    FRAMES_PER_SEC = int(FEAT_SEQ_LEN / CHUNK_LEN_SEC)  # 100

    CHUNK_SIZE = FEAT_SEQ_LEN  # 500
    HOP_SIZE = int(HOP_LEN_SEC * FRAMES_PER_SEC)  # 100

    with torch.no_grad():
        # 注意：这里要求 data_generator 必须以 per_file=True 模式运行
        # 这样 yield 出来的数据才是一个文件的完整切片集合
        for file_cnt, (data, target) in enumerate(data_generator.generate()):

            # --- 1. 数据缝合 (Stitching) ---
            # data 来自 generator: [Num_Chunks, Channels, Seq_Len(500), Freq]
            # 我们需要把它变成: [1, Channels, Total_Time, Freq]

            if isinstance(data, np.ndarray):
                data_batch = torch.tensor(data).float()
            else:
                data_batch = data.float()

            # 维度变换: [B, C, T, F] -> [C, B, T, F] -> [C, B*T, F]
            # 这样就把时间轴接起来了
            C, B, T, F = data_batch.shape[1], data_batch.shape[0], data_batch.shape[2], data_batch.shape[3]

            # permute(1, 0, 2, 3) 把 Channel 提到最前
            # reshape(C, -1, F) 把 Batch 和 Time 合并
            data_continuous = data_batch.permute(1, 0, 2, 3).reshape(C, -1, F)

            # 增加 Batch 维度 -> [1, C, Total_Time, F]
            data_full = data_continuous.unsqueeze(0)

            # 如果 generator 加了 padding (最后一段全是 1e-6)，缝合后末尾会有静音
            # 这不影响滑动窗口，因为模型对静音会预测为背景，或者我们可以截断
            # 这里为了简单，直接用 data_full，只在输出时截断到真实帧数

            T_total = data_full.shape[2]

            # --- 2. 准备 Buffer ---
            NUM_QUERIES = 8
            NUM_CLASSES = params['unique_classes']

            # 动态跑一次 dummy 获取输出的时间缩放比例 (Pooling 因素)
            # 取前 500 帧跑一次
            first_chunk = data_full[:, :, :CHUNK_SIZE, :].to(device)
            out_dummy = model(first_chunk)
            Q_dummy = out_dummy['pred_logits'].shape[1]
            T_out_chunk = Q_dummy // NUM_QUERIES
            scale_factor = T_out_chunk / CHUNK_SIZE

            T_total_out = int(T_total * scale_factor)

            # 初始化全局 Buffer
            global_prob_sum = np.zeros((T_total_out + CHUNK_SIZE, NUM_QUERIES, NUM_CLASSES))
            global_doa_sum = np.zeros((T_total_out + CHUNK_SIZE, NUM_QUERIES, 3))
            overlap_count = np.zeros((T_total_out + CHUNK_SIZE, 1))

            # --- 3. 滑动窗口循环 (Reslicing) ---
            # 现在我们是在 data_full (连续长条) 上滑动，步长由 HOP_SIZE 控制
            for start_frame in range(0, T_total, HOP_SIZE):
                end_frame = start_frame + CHUNK_SIZE

                # 提取切片 (Padding 处理)
                if end_frame > T_total:
                    # 如果不够长，补零
                    pad_len = end_frame - T_total
                    chunk = data_full[:, :, start_frame:, :]
                    chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad_len))
                else:
                    chunk = data_full[:, :, start_frame:end_frame, :]

                chunk = chunk.to(device)

                # 推理
                output = model(chunk)

                # 解析输出
                pred_logits = output['pred_logits']
                pred_doa = output['pred_doa']

                # 恢复形状 [1, T, N, C]
                B_c, Q_c, K_plus_1 = pred_logits.shape
                T_c = Q_c // NUM_QUERIES
                pred_logits = pred_logits.view(B_c, T_c, NUM_QUERIES, K_plus_1)
                pred_doa = pred_doa.view(B_c, T_c, NUM_QUERIES, 3)

                curr_probs = pred_logits.softmax(dim=-1)[0, :, :, :NUM_CLASSES].cpu().numpy()
                curr_doas = pred_doa[0].cpu().numpy()

                # --- 4. 匈牙利匹配与累加 ---
                start_frame_out = int(start_frame * scale_factor)

                for t in range(T_c):
                    global_t = start_frame_out + t

                    p_new = curr_probs[t]
                    d_new = curr_doas[t]

                    cnt = overlap_count[global_t, 0]

                    if cnt == 0:
                        global_prob_sum[global_t] = p_new
                        global_doa_sum[global_t] = d_new
                    else:
                        p_ref = global_prob_sum[global_t] / cnt
                        d_ref = global_doa_sum[global_t] / cnt

                        # Cost Matrix
                        cost_class = np.sum(np.abs(p_new[:, None, :] - p_ref[None, :, :]), axis=-1)
                        cost_doa = np.sum(np.abs(d_new[:, None, :] - d_ref[None, :, :]), axis=-1)
                        C_matrix = cost_class + cost_doa

                        row_ind, col_ind = linear_sum_assignment(C_matrix)

                        p_new_aligned = np.zeros_like(p_new)
                        d_new_aligned = np.zeros_like(d_new)
                        p_new_aligned[col_ind] = p_new[row_ind]
                        d_new_aligned[col_ind] = d_new[row_ind]

                        global_prob_sum[global_t] += p_new_aligned
                        global_doa_sum[global_t] += d_new_aligned

                    overlap_count[global_t] += 1

            # --- 5. 输出生成 ---
            valid_len = min(T_total_out, global_prob_sum.shape[0])
            # 去除 generator 填充的 padding 部分 (如果有必要)
            # data_generator.get_nb_frames_file() 可能不准，依赖于 files_cnt
            # 建议：直接输出全部，评测脚本通常会根据 GT 截断，或者忽略多出的静音

            final_probs = global_prob_sum[:valid_len]
            final_doas = global_doa_sum[:valid_len]
            counts = overlap_count[:valid_len]

            mask = counts > 0
            final_probs[mask[:, 0]] /= counts[mask[:, 0]]
            final_doas[mask[:, 0]] /= counts[mask[:, 0]]

            # 写文件
            current_file_name = test_filelist[file_cnt]
            output_file = os.path.join(dcase_output_folder, current_file_name.replace('.npy', '.csv'))
            output_dict = {}
            CONFIDENCE_THRESHOLD = 0.4

            for t in range(valid_len):
                for n in range(NUM_QUERIES):
                    probs = final_probs[t, n]
                    class_idx = np.argmax(probs)
                    score = probs[class_idx]

                    if score > CONFIDENCE_THRESHOLD:
                        x, y, z = final_doas[t, n]
                        if t not in output_dict: output_dict[t] = []
                        output_dict[t].append([class_idx, x, y, z])

            data_generator.write_output_format_file(output_file, output_dict)

            if params['quick_test'] and nb_test_batches == 4: break
            nb_test_batches += 1

    return 0.0