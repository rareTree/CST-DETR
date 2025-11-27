# Contains routines for labels creation, features extraction and normalization
#


import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plot
import librosa
import time

plot.switch_backend('agg')
import shutil
import math
import wave
import contextlib


# C(n,r)
def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n - r)


class FeatureClass:
    def __init__(self, params, is_eval=False):
        """
        :param params: parameters dictionary
        :param is_eval: if True, does not load dataset labels.
        """

        # Input directories
        self._feat_label_dir = params['feat_label_dir']
        self._dataset_dir = params['dataset_dir']
        self._dataset_combination = '{}_{}'.format(params['dataset'], 'eval' if is_eval else 'dev')
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)

        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')

        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None

        # Local parameters
        self._is_eval = is_eval

        self._fs = params['fs']  # 24000
        self._hop_len_s = params['hop_len_s']  # 0.02s
        self._hop_len = int(self._fs * self._hop_len_s)  # 480

        self._label_hop_len_s = params['label_hop_len_s']  # 0.1s
        self._label_hop_len = int(self._fs * self._label_hop_len_s)  # 2400
        self._label_frame_res = self._fs / float(self._label_hop_len)  # 10帧/秒
        self._nb_label_frames_1s = int(self._label_frame_res)  # 1秒内的标签帧数

        self._win_len = 2 * self._hop_len  # 窗长（采样点数），通常为2倍帧移，此处为960
        self._nfft = self._next_greater_power_of_2(self._win_len)  # 1024

        self._dataset = params['dataset']
        self._eps = 1e-8  # 避免除零的小常数
        self._nb_channels = 4

        self._multi_accdoa = params['multi_accdoa']
        self._use_salsalite = params['use_salsalite']
        self._use_real_imag = params['use_real_imag']  # 是否使用频谱的实部+虚部作为特征

        if self._use_salsalite and self._dataset == 'mic':
            # Initialize the spatial feature constants
            self._lower_bin = np.int(np.floor(params['fmin_doa_salsalite'] * self._nfft / np.float(self._fs)))
            self._lower_bin = np.max((1, self._lower_bin))
            self._upper_bin = np.int(
                np.floor(np.min((params['fmax_doa_salsalite'], self._fs // 2)) * self._nfft / np.float(self._fs)))

            # Normalization factor for salsalite
            c = 343
            self._delta = 2 * np.pi * self._fs / (self._nfft * c)
            self._freq_vector = np.arange(self._nfft // 2 + 1)
            self._freq_vector[0] = 1
            self._freq_vector = self._freq_vector[None, :, None]  # 1 x n_bins x 1

            # Initialize spectral feature constants
            self._cutoff_bin = np.int(np.floor(params['fmax_spectra_salsalite'] * self._nfft / np.float(self._fs)))
            assert self._upper_bin <= self._cutoff_bin, 'Upper bin for doa featurei {} is higher than cutoff bin for spectrogram {}!'.format()
            self._nb_mel_bins = self._cutoff_bin - self._lower_bin
        else:
            self._nb_mel_bins = params['nb_mel_bins']
            self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T
            self._nb_freq_bins = self._nfft // 2 + 1  # 513
        # Sound event classes dictionary
        self._nb_unique_classes = params['unique_classes']

        self._filewise_frames = {}  # 存储每个文件的特征帧和标签帧数量

    def get_frame_stats(self):
        """计算每个音频文件的特征帧和标签帧数量，存储到_filewise_frames"""
        if len(self._filewise_frames) != 0:
            return

        print('Computing frame stats:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))
        for sub_folder in os.listdir(self._aud_dir):
            loc_aud_folder = os.path.join(self._aud_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_aud_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                with contextlib.closing(wave.open(os.path.join(loc_aud_folder, wav_filename), 'r')) as f:
                    audio_len = f.getnframes()
                nb_feat_frames = int(audio_len / float(self._hop_len))
                nb_label_frames = int(audio_len / float(self._label_hop_len))
                self._filewise_frames[file_name.split('.')[0]] = [nb_feat_frames, nb_label_frames]
        return

    def _load_audio(self, audio_path):
        """加载音频文件并归一化"""
        fs, audio = wav.read(audio_path)  # 读取音频（采样率fs，音频数据audio）
        # 归一化到[-1, 1]范围（WAV文件通常为16位整数，最大值32768），并添加小常数避免零
        audio = audio[:, :self._nb_channels] / 32768.0 + self._eps
        return audio, fs

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        """计算大于x的最小2的幂（加速FFT）"""
        return 2 ** (x - 1).bit_length()

    def _STFT(self, audio_input, _nb_frames):
        """
                计算多通道音频的短时傅里叶变换（STFT）
                :param audio_input: 音频数据 [采样点数, 通道数]
                :param _nb_frames: 期望的特征帧数
                :return: 频谱 [帧数, 频率bins, 通道数]
        """
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2  # 512
        spectra = []
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft,
                                        hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra.append(stft_ch[:, :_nb_frames])
        return np.array(spectra).T   # 转置为 [帧数, 频率bins, 通道数]

    def _ISTFT(self, spect_input, audio_length):
        """逆STFT（将频谱转换回音频）"""
        _nb_ch = spect_input.shape[-1]
        nb_bins = self._nfft // 2
        audio = []
        for ch_cnt in range(_nb_ch):
            audio_ch = librosa.core.istft(np.asfortranarray(spect_input[:, :, ch_cnt].T), hop_length=self._hop_len,
                                          win_length=self._win_len, window='hann', length=audio_length)
            audio.append(audio_ch)
        return np.array(audio).T  # 转置为 [采样点数, 通道数]

    def _get_mel_spectrogram(self, linear_spectra):
        """
               将线性频谱转换为梅尔频谱（用于音频的频谱特征）
               :param linear_spectra: 线性频谱 [帧数, 频率bins, 通道数]
               :return: 梅尔频谱特征 [帧数, 通道数*梅尔bins]
        """
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            # 计算功率谱（幅度平方）
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt]) ** 2
            # 应用梅尔滤波器组
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            # 转换为分贝值（log压缩）
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat

    def _get_foa_intensity_vectors(self, linear_spectra):
        """
                计算FOA（一阶Ambisonics）强度向量（空间特征，用于声源定位）
                :param linear_spectra: 线性频谱 [帧数, 频率bins, 4]（W, X, Y, Z通道）
                :return: 强度向量特征 [帧数, 3*梅尔bins]（x, y, z三个方向）
        """
        W = linear_spectra[:, :, 0]  # W通道（全向分量）
        # 计算强度向量（I = Re{W*·X, W*·Y, W*·Z}，其中*为共轭）
        I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
        # 能量归一化（E = |W|² + (|X|² + |Y|² + |Z|²)/3）
        E = self._eps + (np.abs(W) ** 2 + ((np.abs(linear_spectra[:, :, 1:]) ** 2).sum(-1)) / 3.0)

        I_norm = I / E[:, :, np.newaxis]  # 归一化强度向量
        # 转换到梅尔频带
        I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0, 2, 1)), self._mel_wts), (0, 2, 1))
        # 维度调整：[帧数, 梅尔bins, 3] -> [帧数, 3*梅尔bins]
        foa_iv = I_norm_mel.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], self._nb_mel_bins * 3))
        if np.isnan(foa_iv).any():
            print('Feature extraction is generating nan outputs')
            exit()
        return foa_iv

    def _reshape_linear_spectra(self, linear_spectra):
        """将频谱的实部和虚部合并为特征"""
        # [T,F,C] -> [T,2C,F] -> [T, 2C*F]
        # reshaped = linear_spectra.transpose((0,2,1,)).reshape((linear_spectra.shape[0]), linear_spectra.shape[1]* linear_spectra.shape[2])
        reshaped = linear_spectra.transpose((0, 2, 1,))  # [T,C,F]
        reshaped = self._get_linear_spect_real_imag(reshaped)  # [T,C,F] -> [T,2C,F]
        reshaped = reshaped.reshape((linear_spectra.shape[0]), linear_spectra.shape[1] * linear_spectra.shape[2])  #[T, 2C*F]
        return reshaped

    def _get_linear_spect_real_imag(self, linear_spectra):
        """分离频谱的实部和虚部并拼接"""
        real_feat = linear_spectra.real
        imag_feat = linear_spectra.imag
        feat = np.concatenate([real_feat, imag_feat], axis=1)  ## 拼接为[..., 2C, ...]
        return feat

    def _get_gcc(self, linear_spectra):
        """
               计算广义互相关（GCC）特征（用于麦克风阵列的声源定位）
               :param linear_spectra: 线性频谱 [帧数, 频率bins, 通道数]
               :return: GCC特征 [帧数, 通道对数量*梅尔bins]
        """
        gcc_channels = nCr(linear_spectra.shape[-1], 2)  # 通道对数量（C(4,2)=6）
        gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
        cnt = 0
        # 遍历所有通道对（m < n）
        for m in range(linear_spectra.shape[-1]):
            for n in range(m + 1, linear_spectra.shape[-1]):
                # 计算互功率谱（R = X_m*·X_n，*为共轭）
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                # 相位变换（PHAT加权：只保留相位信息）
                cc = np.fft.irfft(np.exp(1.j * np.angle(R)))
                # 截取有效范围（对称中心附近的梅尔bins）
                cc = np.concatenate((cc[:, -self._nb_mel_bins // 2:], cc[:, :self._nb_mel_bins // 2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        # 维度调整：[帧数, 梅尔bins, 通道对] -> [帧数, 通道对*梅尔bins]
        return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    def _get_salsalite(self, linear_spectra):
        """
                提取SALSA-Lite特征（结合频谱和相位特征，用于空间定位）
                :param linear_spectra: 线性频谱 [帧数, 频率bins, 通道数]
                :return: 拼接后的SALSA-Lite特征
        """
        # Adapted from the official SALSA repo- https://github.com/thomeou/SALSA
        # spatial features
        # 1. 空间特征（相位转换为距离）
        # 计算通道1-3与通道0的相位差（用于DOA估计）
        phase_vector = np.angle(linear_spectra[:, :, 1:] * np.conj(linear_spectra[:, :, 0, None]))
        # 相位差转换为距离（基于声速和频率）
        phase_vector = phase_vector / (self._delta * self._freq_vector)
        # 截取有效频率范围并清零高频部分
        phase_vector = phase_vector[:, self._lower_bin:self._cutoff_bin, :]
        phase_vector[:, self._upper_bin:, :] = 0
        # 维度调整：[帧数, 频率bins, 3] -> [帧数, 3*(cutoff-lower)]
        phase_vector = phase_vector.transpose((0, 2, 1)).reshape((phase_vector.shape[0], -1))

        # spectral features
        # 2. 频谱特征（功率谱的分贝值）
        linear_spectra = np.abs(linear_spectra) ** 2  # 功率谱
        for ch_cnt in range(linear_spectra.shape[-1]):
            # 转换为分贝值
            linear_spectra[:, :, ch_cnt] = librosa.power_to_db(linear_spectra[:, :, ch_cnt], ref=1.0, amin=1e-10,
                                                               top_db=None)
        # 截取有效频率范围
        linear_spectra = linear_spectra[:, self._lower_bin:self._cutoff_bin, :]
        # 维度调整：[帧数, 频率bins, 通道数] -> [帧数, 通道数*(cutoff-lower)]
        linear_spectra = linear_spectra.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        # 拼接频谱和空间特征
        return np.concatenate((linear_spectra, phase_vector), axis=-1)

    def _get_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(audio_filename)

        nb_feat_frames = int(len(audio_in) / float(self._hop_len))
        nb_label_frames = int(len(audio_in) / float(self._label_hop_len))
        self._filewise_frames[os.path.basename(audio_filename).split('.')[0]] = [nb_feat_frames, nb_label_frames]

        audio_spec = self._STFT(audio_in, nb_feat_frames)
        return audio_spec

    # OUTPUT LABELS
    def get_labels_for_file(self, _desc_file, _nb_label_frames):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels
        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 3*max_classes], max_classes each for x, y, z axis,
        """

        # If using Hungarian net set default DOA value to a fixed value greater than 1 for all axis. We are choosing a fixed value of 10
        # If not using Hungarian net use a deafult DOA, which is a unit vector. We are choosing (x, y, z) = (0, 0, 1)
        # 初始化标签数组
        se_label = np.zeros((_nb_label_frames, self._nb_unique_classes))   # SED标签（是否存在事件）
        x_label = np.zeros((_nb_label_frames, self._nb_unique_classes))  # X方向DOA
        y_label = np.zeros((_nb_label_frames, self._nb_unique_classes))  # Y方向DOA
        z_label = np.zeros((_nb_label_frames, self._nb_unique_classes))  # Z方向DOA

        # 遍历描述文件中的每帧标签
        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:  # 只处理有效帧
                for active_event in active_event_list:
                    se_label[frame_ind, active_event[0]] = 1
                    x_label[frame_ind, active_event[0]] = active_event[2]
                    y_label[frame_ind, active_event[0]] = active_event[3]
                    z_label[frame_ind, active_event[0]] = active_event[4]

        # 拼接SED和DOA标签为一个矩阵
        label_mat = np.concatenate((se_label, x_label, y_label, z_label), axis=1)
        return label_mat

    # OUTPUT LABELS
    def get_adpit_labels_for_file(self, _desc_file, _nb_label_frames):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels
        for multi-ACCDOA with Auxiliary Duplicating Permutation Invariant Training (ADPIT)
        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 6, 4(=act+XYZ), max_classes]
        """

        se_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))  # [nb_frames, 6, max_classes]
        x_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:  # 只处理有效帧
                active_event_list.sort(key=lambda x: x[0])  # 按类别ID排序（便于同类别分组）
                active_event_list_per_class = []  # 存储当前类别的事件列表
                for i, active_event in enumerate(active_event_list):
                    active_event_list_per_class.append(active_event)  # 添加到当前类别列表
                    # 处理最后一个事件
                    if i == len(active_event_list) - 1:  # if the last
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]

                    elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                        active_event_list_per_class = []

        label_mat = np.stack((se_label, x_label, y_label, z_label), axis=2)  # [nb_frames, 6, 4(=act+XYZ), max_classes]
        return label_mat

    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------

    def extract_file_feature(self, _arg_in):
        _file_cnt, _wav_path, _feat_path = _arg_in
        spect = self._get_spectrogram_for_file(_wav_path)

        # extract mel
        if not self._use_salsalite and not self._use_real_imag:
            mel_spect = self._get_mel_spectrogram(spect)

        feat = None
        if self._dataset == 'foa':
            # extract intensity vectors
            if self._use_real_imag:
                feat = self._reshape_linear_spectra(spect)  # [T,F,C] -> [T,C,F,2] -> [T,2C,F] -> [T,2C*F]
            else:
                foa_iv = self._get_foa_intensity_vectors(spect)
                feat = np.concatenate((mel_spect, foa_iv), axis=-1)  # [T,C*F]

        elif self._dataset == 'mic':
            if self._use_salsalite:
                feat = self._get_salsalite(spect)
            else:
                # extract gcc
                gcc = self._get_gcc(spect)
                feat = np.concatenate((mel_spect, gcc), axis=-1)
        else:
            print('ERROR: Unknown dataset format {}'.format(self._dataset))
            exit()

        if feat is not None:
            print('{}: {}, {}'.format(_file_cnt, os.path.basename(_wav_path), feat.shape))
            np.save(_feat_path, feat)

    def extract_all_feature(self):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir()   # 获取未归一化特征目录
        create_folder(self._feat_dir)  # 创建目录

        start_s = time.time()
        # extraction starts
        print('Extracting spectrogram:')
        arg_list = []
        if not self._is_eval:  # 训练模式（音频在子文件夹中）
            print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
                self._aud_dir, self._desc_dir, self._feat_dir))
            for sub_folder in os.listdir(self._aud_dir):
                loc_aud_folder = os.path.join(self._aud_dir, sub_folder)
                for file_cnt, file_name in enumerate(os.listdir(loc_aud_folder)):
                    print(file_name)
                    wav_filename = '{}.wav'.format(file_name.split('.')[0])
                    wav_path = os.path.join(loc_aud_folder, wav_filename)
                    feat_path = os.path.join(self._feat_dir, '{}.npy'.format(wav_filename.split('.')[0]))
                    self.extract_file_feature((file_cnt, wav_path, feat_path))
                    arg_list.append((file_cnt, wav_path, feat_path))

        else:  # 评估模式（音频直接在目录中）
            print('\t\taud_dir {}\n\t\tfeat_dir {}'.format(
                self._aud_dir, self._feat_dir))
            for file_cnt, file_name in enumerate(os.listdir(self._aud_dir)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                wav_path = os.path.join(self._aud_dir, wav_filename)
                feat_path = os.path.join(self._feat_dir, '{}.npy'.format(wav_filename.split('.')[0]))
                self.extract_file_feature((file_cnt, wav_path, feat_path))
                arg_list.append((file_cnt, wav_path, feat_path))
        print(time.time() - start_s)

    def preprocess_features(self):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir()  # 未归一化特征目录
        self._feat_dir_norm = self.get_normalized_feat_dir()  # 归一化特征目录
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()  # 归一化模型保存路径
        spec_scaler = None  # 标准化器

        # pre-processing starts
        if self._is_eval:
            spec_scaler = joblib.load(normalized_features_wts_file)  # 评估模式：加载训练好的标准化器
            print('Normalized_features_wts_file: {}. Loaded.'.format(normalized_features_wts_file))

        else:  # 训练/开发模式：计算标准化参数（均值和方差）
            print('Estimating weights for normalizing feature files:')
            print('\t\tfeat_dir: {}'.format(self._feat_dir))

            spec_scaler = preprocessing.StandardScaler()
            # 遍历所有特征文件，累积计算均值和方差
            for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print('{}: {}'.format(file_cnt, file_name))
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                spec_scaler.partial_fit(feat_file)  # 增量拟合
                del feat_file  # 释放内存
            # 保存标准化器
            joblib.dump(
                spec_scaler,
                normalized_features_wts_file
            )
            print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

        # 对所有特征文件进行归一化并保存
        print('Normalizing feature files:')
        print('\t\tfeat_dir_norm {}'.format(self._feat_dir_norm))
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self._feat_dir, file_name))
            feat_file = spec_scaler.transform(feat_file)  # 标准化
            np.save(
                os.path.join(self._feat_dir_norm, file_name),
                feat_file
            )
            del feat_file  # 释放内存

        print('normalized files written to {}'.format(self._feat_dir_norm))

    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self.get_frame_stats()  # 先计算帧统计信息
        self._label_dir = self.get_label_dir()  # 获取标签保存目录

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        create_folder(self._label_dir)
        for sub_folder in os.listdir(self._desc_dir):
            loc_desc_folder = os.path.join(self._desc_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_desc_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                # 检查该文件是否在帧统计中
                if file_name.split('.')[0] in self._filewise_frames.keys():
                    # 获取该文件的标签帧数
                    nb_label_frames = self._filewise_frames[file_name.split('.')[0]][1]
                    # 加载描述文件（极坐标格式）
                    desc_file_polar = self.load_output_format_file(os.path.join(loc_desc_folder, file_name))
                    # 转换为笛卡尔坐标（x,y,z）
                    desc_file = self.convert_output_format_polar_to_cartesian(desc_file_polar)
                    if self._multi_accdoa:
                        label_mat = self.get_adpit_labels_for_file(desc_file, nb_label_frames)
                    else:
                        label_mat = self.get_labels_for_file(desc_file, nb_label_frames)
                    print('{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
                    np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

    # -------------------------------  DCASE OUTPUT  FORMAT FUNCTIONS -------------------------------
    def load_output_format_file(self, _output_format_file):
        """
        Loads DCASE output format csv file and returns it in dictionary format
        :param _output_format_file: DCASE output format CSV
        :return: _output_dict: dictionary
        """
        """
        加载DCASE输出格式的CSV标签文件，转换为字典格式
        :param _output_format_file: DCASE格式CSV文件路径
        :return: _output_dict: 字典 {帧索引: 事件列表}，事件格式：[类别ID, ..., 坐标]
        """
        _output_dict = {}
        _fid = open(_output_format_file, 'r')
        # next(_fid)
        for _line in _fid:
            _words = _line.strip().split(',')  # 按逗号分割行
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 4:  # frame, class idx, polar coordinates(2) # no distance data
                _words[1], _words[2], _words[3] = float(_words[1]), float(_words[2]), float(_words[3])
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3])])
            if len(
                    _words) == 5:  # frame, class idx, source_id, polar coordinates(2) # no distance data, for example in synthetic data fold 1 and 2
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
            if len(_words) == 6:  # frame, class idx, source_id, polar coordinates(2), distance
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
            elif len(_words) == 7:  # frame, class idx, source_id, cartesian coordinates(3), distance
                _output_dict[_frame_ind].append(
                    [int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
        _fid.close()
        return _output_dict

    def write_output_format_file(self, _output_format_file, _output_format_dict, is_eval=False):
        """
        Writes DCASE output format csv file, given output format dictionary
        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count and distance we use fixed values.
                if not is_eval:
                    _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]),
                                                               float(_value[2]), float(_value[3]), 0))
                else:
                    _fid.write(
                        '{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), float(_value[1]), float(_value[2])))
        _fid.close()

    def segment_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in segments of length 1s from reference dataset
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each segment of audio
                dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
        '''
        '''
                将帧级标签按1秒分段，收集每段内的类别-位置信息
                :param _pred_dict: 帧级标签字典 {帧索引: 事件列表}
                :param _max_frames: 总帧数
                :return: 分段标签字典 {段索引: {类别ID: [帧内索引, 方位角, 仰角]}}
                '''
        # 计算总段数（1秒一段）
        nb_blocks = int(np.ceil(_max_frames / float(self._nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}  # 初始化分段字典
        for frame_cnt in range(0, _max_frames, self._nb_label_frames_1s):

            # Collect class-wise information for each block
            # [class][frame] = <list of doa values>
            # Data structure supports multi-instance occurence of same class
            block_cnt = frame_cnt // self._nb_label_frames_1s  # 当前段索引
            loc_dict = {}  # 存储当前段内的类别-帧-位置信息
            for audio_frame in range(frame_cnt, frame_cnt + self._nb_label_frames_1s):
                if audio_frame not in _pred_dict:
                    continue
                # 遍历当前帧的所有事件
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict:
                        loc_dict[value[0]] = {}  # 初始化该类别的帧字典
                    # 计算帧在段内的索引
                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]:
                        loc_dict[value[0]][block_frame] = []  # 初始化该帧的位置列表
                    loc_dict[value[0]][block_frame].append(value[1:])   # 添加位置信息

            # Update the block wise details collected above in a global structure
            # 将当前段的信息更新到全局字典
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_cnt] = []
                # 提取帧索引和位置信息并存储
                keys = [k for k in loc_dict[class_cnt]]
                values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

                output_dict[block_cnt][class_cnt].append([keys, values])

        return output_dict

    def regression_label_format_to_output_format(self, _sed_labels, _doa_labels):
        """
        Converts the sed (classification) and doa labels predicted in regression format to dcase output format.
        :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
        :param _doa_labels: DOA labels matrix [nb_frames, 2*nb_classes] or [nb_frames, 3*nb_classes]
        :return: _output_dict: returns a dict containing dcase output format
        """

        _nb_classes = self._nb_unique_classes
        _is_polar = _doa_labels.shape[-1] == 2 * _nb_classes
        _azi_labels, _ele_labels = None, None
        _x, _y, _z = None, None, None
        if _is_polar:
            _azi_labels = _doa_labels[:, :_nb_classes]
            _ele_labels = _doa_labels[:, _nb_classes:]
        else:
            _x = _doa_labels[:, :_nb_classes]
            _y = _doa_labels[:, _nb_classes:2 * _nb_classes]
            _z = _doa_labels[:, 2 * _nb_classes:]

        _output_dict = {}
        for _frame_ind in range(_sed_labels.shape[0]):
            _tmp_ind = np.where(_sed_labels[_frame_ind, :])
            if len(_tmp_ind[0]):
                _output_dict[_frame_ind] = []
                for _tmp_class in _tmp_ind[0]:
                    if _is_polar:
                        _output_dict[_frame_ind].append(
                            [_tmp_class, _azi_labels[_frame_ind, _tmp_class], _ele_labels[_frame_ind, _tmp_class]])
                    else:
                        _output_dict[_frame_ind].append(
                            [_tmp_class, _x[_frame_ind, _tmp_class], _y[_frame_ind, _tmp_class],
                             _z[_frame_ind, _tmp_class]])
        return _output_dict

    def convert_output_format_polar_to_cartesian(self, in_dict):
        """将极坐标（方位角、仰角）标签转换为笛卡尔坐标（x,y,z）"""
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    # tmp_val格式：[类别ID, 源ID, 方位角(度), 仰角(度)]
                    ele_rad = tmp_val[3] * np.pi / 180.  # 仰角转为弧度
                    azi_rad = tmp_val[2] * np.pi / 180  # 方位角转为弧度

                    # 极坐标转笛卡尔坐标（单位球面上）
                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
        return out_dict

    def convert_output_format_cartesian_to_polar(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]

                    # in degrees
                    azimuth = np.rint(np.arctan2(y, x) * 180 / np.pi)
                    elevation = np.rint(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi)
                    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                    out_dict[frame_cnt].append([tmp_val[0], 0, azimuth, elevation])
        return out_dict

    def convert_output_format_cartesian_to_polar_eval(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    x, y, z = tmp_val[1], tmp_val[2], tmp_val[3]

                    # in degrees
                    azimuth = np.rint(np.arctan2(y, x) * 180 / np.pi)
                    elevation = np.rint(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi)

                    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                    out_dict[frame_cnt].append([tmp_val[0], azimuth, elevation])
        return out_dict

    # ------------------------------- Misc public functions -------------------------------
    def get_normalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_norm'.format('{}_salsa'.format(self._dataset_combination) if (
                    self._dataset == 'mic' and self._use_salsalite) else self._dataset_combination)
        )

    def get_unnormalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}'.format('{}_salsa'.format(self._dataset_combination) if (
                    self._dataset == 'mic' and self._use_salsalite) else self._dataset_combination)
        )

    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            return os.path.join(
                self._feat_label_dir,
                '{}_label'.format(
                    '{}_adpit'.format(self._dataset_combination) if self._multi_accdoa else self._dataset_combination)
            )

    def get_normalized_wts_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_wts'.format(self._dataset)
        )

    def get_nb_channels(self):
        return self._nb_channels

    def get_nb_classes(self):
        return self._nb_unique_classes

    def nb_frames_1s(self):
        return self._nb_label_frames_1s

    def get_hop_len_sec(self):
        return self._hop_len_s

    def get_nb_mel_bins(self):
        return self._nb_mel_bins

    def get_nb_linear_bins(self):
        return self._nb_freq_bins


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)


def delete_and_create_folder(folder_name):
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)
