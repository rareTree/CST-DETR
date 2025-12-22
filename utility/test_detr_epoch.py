# utility/test_detr_epoch.py
import os
import torch
import numpy as np


def test_detr_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    test_filelist = data_generator.get_filelist()
    nb_test_batches, test_loss = 0, 0.
    model.eval()

    with torch.no_grad():
        for file_cnt, (data, target) in enumerate(data_generator.generate()):
            data = torch.tensor(data).float().to(device)
            target = torch.tensor(target).float().to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            nb_test_batches += 1

            pred_logits = output['pred_logits']
            pred_doa = output['pred_doa']

            B, T, N, K_plus_1 = pred_logits.shape
            K = params['unique_classes']

            # --- 最终策略: 极低阈值 ---
            # 趋势显示 0.4 比 0.5 好，说明我们需要更高的 Recall
            # 这一把直接试 0.35，冲击 0.38 大关
            pred_scores = pred_logits.softmax(dim=-1)
            confidence_threshold = 0.35

            max_probs, pred_classes = pred_scores.max(dim=-1)
            bg_class_idx = params['unique_classes']
            pred_classes[max_probs < confidence_threshold] = bg_class_idx

            # 重组与写入
            pred_classes_file = pred_classes.reshape(B * T, N).cpu().numpy()
            pred_doa_file = pred_doa.reshape(B * T, N, 3).cpu().numpy()

            current_file_name = test_filelist[file_cnt]
            output_file = os.path.join(dcase_output_folder, current_file_name.replace('.npy', '.csv'))
            output_dict = {}

            num_frames = B * T
            for frame_cnt in range(num_frames):
                for n_query in range(N):
                    class_idx = pred_classes_file[frame_cnt, n_query]
                    if class_idx < K:
                        doa_vector = pred_doa_file[frame_cnt, n_query]
                        x, y, z = doa_vector[0], doa_vector[1], doa_vector[2]
                        if frame_cnt not in output_dict:
                            output_dict[frame_cnt] = []
                        output_dict[frame_cnt].append([class_idx, x, y, z])

            data_generator.write_output_format_file(output_file, output_dict)

            if params['quick_test'] and nb_test_batches == 4: break

        test_loss /= nb_test_batches
    return test_loss