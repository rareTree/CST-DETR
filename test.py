import torch
import torch.nn as nn
from einops import rearrange
import scipy.io.wavfile as wav
import librosa
import numpy as np
import os

_nb_channels = 4
_eps = 1e-8
_hop_len = 480
_label_hop_len = 2400
_nfft = 1024
_win_len = 960
audio_path = './data/2023DCASE_data/foa_dev/dev-train-sony/fold3_room21_mix002.wav'
_output_format_file = './data/2023DCASE_data/metadata_dev/dev-train-sony/fold3_room21_mix002.csv'
# fs, audio = wav.read(audio_path)  # 读取音频（采样率fs，音频数据audio）
# # 归一化到[-1, 1]范围（WAV文件通常为16位整数，最大值32768），并添加小常数避免零
# print(fs)
# print(audio.shape)
# audio = audio[:, :_nb_channels] / 32768.0 + _eps
# print(audio.shape)
#
# nb_feat_frames = int(len(audio) / float(_hop_len))
# nb_label_frames = int(len(audio) / float(_label_hop_len))
# print(nb_feat_frames, nb_label_frames)
#
# _nb_ch = audio.shape[1]
# nb_bins = _nfft // 2  # 512
# print(_nb_ch, nb_bins)
# spectra = []
# for ch_cnt in range(_nb_ch):
#     stft_ch = librosa.core.stft(np.asfortranarray(audio[:, ch_cnt]), n_fft=_nfft,
#                                 hop_length=_hop_len,
#                                 win_length=_win_len, window='hann')
#     print(stft_ch.shape)
#     spectra.append(stft_ch[:, :nb_feat_frames])
# print(np.array(spectra).T.shape)

# _output_dict = {}
# _fid = open(_output_format_file, 'r')
# # next(_fid)
# for _line in _fid:
#     _words = _line.strip().split(',')  # 按逗号分割行
#     _frame_ind = int(_words[0])
#     if _frame_ind not in _output_dict:
#         _output_dict[_frame_ind] = []
#     if len(_words) == 4:  # frame, class idx, polar coordinates(2) # no distance data
#         _words[1], _words[2], _words[3] = float(_words[1]), float(_words[2]), float(_words[3])
#         _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3])])
#     if len(_words) == 5:  # frame, class idx, source_id, polar coordinates(2) # no distance data, for example in synthetic data fold 1 and 2
#          _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
#     if len(_words) == 6:  # frame, class idx, source_id, polar coordinates(2), distance
#          _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
#     elif len(_words) == 7:  # frame, class idx, source_id, cartesian coordinates(3), distance
#          _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
# _fid.close()
# print(_output_dict)
# out_dict = {}
# for frame_cnt in _output_dict.keys():
#     if frame_cnt not in out_dict:
#         out_dict[frame_cnt] = []
#         for tmp_val in _output_dict[frame_cnt]:
#             # tmp_val格式：[类别ID, 源ID, 方位角(度), 仰角(度)]
#             ele_rad = tmp_val[3] * np.pi / 180.  # 仰角转为弧度
#             azi_rad = tmp_val[2] * np.pi / 180  # 方位角转为弧度
#
#             # 极坐标转笛卡尔坐标（单位球面上）
#             tmp_label = np.cos(ele_rad)
#             x = np.cos(azi_rad) * tmp_label
#             y = np.sin(azi_rad) * tmp_label
#             z = np.sin(ele_rad)
#             print(f"ele_rad, azi_red: {ele_rad} {azi_rad}, x, y, z: {x}, {y}, {z}")
#             out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
#
# print(out_dict)
# _nb_label_frames = 7000
# _nb_unique_classes = 13
# se_label = np.zeros((_nb_label_frames, _nb_unique_classes))
# x_label = np.zeros((_nb_label_frames, _nb_unique_classes))  # X方向DOA
# y_label = np.zeros((_nb_label_frames, _nb_unique_classes))  # Y方向DOA
# z_label = np.zeros((_nb_label_frames, _nb_unique_classes))  # Z方向DOA
# print(se_label.shape)
# label_mat = np.concatenate((se_label, x_label, y_label, z_label), axis=1)
# print(label_mat.shape)
#
# for frame_ind, active_event_list in out_dict.items():
#     if frame_ind < _nb_label_frames:  # 只处理有效帧
#         active_event_list.sort(key=lambda x: x[0])  # 按类别ID排序（便于同类别分组）
#         print(frame_ind,active_event_list)
# _is_eval = False
# _feat_dir = './data/feature_labels_2023/foa_dev_norm'
# filename = 'fold3_room4_mix001.npy'
# # for filename in os.listdir(_feat_dir):
# #     if not _is_eval:
# #         # if int(filename[4]) in_splits:  # check which split the file belongs to
# #         print(filename[4])
# temp_feat = np.load(os.path.join(_feat_dir, filename))
# print(temp_feat.shape[0] % 250)



