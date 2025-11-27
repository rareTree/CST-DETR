#
# Data generator for training of the SELD models
#

import os
import numpy as np
import cls.cls_feature_class as cls_feature_class
from collections import deque
import random

class DataGenerator(object):
    def __init__(
            self, params, split=1, shuffle=True, per_file=False, is_eval=False
    ):
        self.DETR = False
        self._per_file = per_file  # 是否按单个文件生成批次（用于评估，确保每个文件的完整输出）
        self._is_eval = is_eval   # 是否为评估模式（True：评估，False：训练/验证）
        self._splits = np.array(split)  # 数据集分割（如训练集/验证集/测试集的标识）
        self._batch_size = params['batch_size']
        self._feature_seq_len = params['feature_sequence_length']  # 每个样本的特征序列长度
        self._label_seq_len = params['label_sequence_length']  # 每个样本的标签序列长度
        self._shuffle = shuffle  # 是否打乱数据顺序
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)

        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()

        self._multi_accdoa = params['multi_accdoa']
        self._use_real_imag = params['use_real_imag']

        self._filenames_list = list()

        self._nb_frames_file = 0  # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        if self._use_real_imag:
            self._nb_f_bins = self._feat_cls.get_nb_linear_bins()
        else:
            self._nb_f_bins = self._feat_cls.get_nb_mel_bins()

        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None  # DOA label length
        self._nb_classes = self._feat_cls.get_nb_classes()

        self._circ_buf_feat = None  # 特征循环缓冲区（用于拼接多文件数据）
        self._circ_buf_label = None  # 标签循环缓冲区

        self._get_filenames_list_and_feat_label_sizes()

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list), self._nb_classes,
                self._nb_frames_file, self._nb_f_bins, self._nb_ch, self._label_len
            )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                params['dataset'], split,
                self._batch_size, self._feature_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir
            )
        )

    def _get_filenames_list_and_feat_label_sizes(self):
        print('Computing some stats about the dataset')
        self.max_frames, self.total_frames, temp_feat = -1, 0, []  # 最大帧数、总帧数、临时特征存储

        # 遍历特征目录下的所有文件，筛选符合条件的文件
        for filename in os.listdir(self._feat_dir):
            if not self._is_eval:
                # 训练/验证模式：只保留属于指定分割的文件（通过文件名判断分割标识）
                if int(filename[4]) in self._splits:  # check which split the file belongs to
                    temp_feat = self._get_filelist_n_frames(filename)
            else:
                temp_feat = self._get_filelist_n_frames(filename)

        if len(temp_feat) != 0:
            # 若按文件生成批次（per_file），则每文件帧数为最大帧数；否则为单个文件的实际帧数
            self._nb_frames_file = self.max_frames if self._per_file else temp_feat.shape[0]
            # 计算通道数：特征维度 = 通道数 × 频率bins → 通道数 = 特征维度 // 频率bins
            self._nb_ch = temp_feat.shape[1] // self._nb_f_bins
        else:
            print('Loading features failed')
            exit()

        if not self._is_eval:
            # 训练/验证模式：加载第一个标签文件，确定标签维度
            temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
            if self._multi_accdoa is True:
                # 多轨迹ACCDOA：标签维度为（轨迹数，轴数，类别数）
                self._num_track_dummy = temp_label.shape[-3]
                self._num_axis = temp_label.shape[-2]
                self._num_class = temp_label.shape[-1]
            else:
                self._label_len = temp_label.shape[-1]
            self._doa_len = 3  # Cartesian  # DOA使用三维笛卡尔坐标（x,y,z）

        if self._per_file:
            # 按文件生成批次时：批次大小调整为单个文件所需的批次数（向上取整）
            self._batch_size = int(np.ceil(self.max_frames / float(self._feature_seq_len)))
            print(
                '\tWARNING: Resetting batch size to {}. To accommodate the inference of longest file of {} frames in a single batch'.format(
                    self._batch_size, self.max_frames))
            self._nb_total_batches = len(self._filenames_list)  # 总批次数 = 文件数（每个文件1批）
        else:
            # 不按文件生成批次时：总批次数 = 总帧数 //（批次大小 × 特征序列长度）
            self._nb_total_batches = int(np.floor(self.total_frames / (self._batch_size * self._feature_seq_len)))

        self._feature_batch_seq_len = self._batch_size * self._feature_seq_len
        self._label_batch_seq_len = self._batch_size * self._label_seq_len

        return

    def generate(self):
        """
        Generates batches of samples
        :return:
        """
        if self._shuffle:
            random.shuffle(self._filenames_list)  # 打乱文件名顺序（仅训练时）

        # Ideally this should have been outside the while loop. But while generating the test data we want the data
        # to be the same exactly for all epoch's hence we keep it here.
        self._circ_buf_feat = deque()  # 特征缓冲区（双端队列，高效append/popleft）
        self._circ_buf_label = deque()  # 标签缓冲区

        file_cnt = 0
        if self._is_eval:
            # 评估模式：只生成特征（无标签）
            for i in range(self._nb_total_batches):
                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                # 循环填充缓冲区，直到至少有1批特征的帧数
                while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                    # 加载当前文件的特征
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                    for row_cnt, row in enumerate(temp_feat):
                        # 将特征逐行添加到缓冲区
                        self._circ_buf_feat.append(row)

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6
            
                        for row_cnt, row in enumerate(extra_feat):
                            self._circ_buf_feat.append(row)

                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_f_bins * self._nb_ch))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()  # 从缓冲区左侧取出数据
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_f_bins))

                # Split to sequences
                # 将特征拆分为多个序列（每个序列长度为_feature_seq_len）
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                # 转置维度：（批次大小，通道数，序列长度，频率bins）→ 适配模型输入
                feat = np.transpose(feat, (0, 2, 1, 3))

                yield feat  # 返回特征批次

        else:
            for i in range(self._nb_total_batches):

                # load feat and label to circular buffer. Always maintain at least one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                    # if not self._SS_multitask:
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                    temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))

                    if not self._per_file:
                        # Inorder to support variable length features, and labels of different resolution.
                        # We remove all frames in features and labels matrix that are outside
                        # the multiple of self._label_seq_len and self._feature_seq_len. Further we do this only in training.
                        temp_label = temp_label[:temp_label.shape[0] - (temp_label.shape[0] % self._label_seq_len)]
                        temp_mul = temp_label.shape[0] // self._label_seq_len
                        temp_feat = temp_feat[:temp_mul * self._feature_seq_len, :]

                    for f_row in temp_feat:
                        self._circ_buf_feat.append(f_row)
                    for l_row in temp_label:
                        self._circ_buf_label.append(l_row)

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6

                        label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                        if self._multi_accdoa is True:
                            extra_labels = np.zeros(
                                (label_extra_frames, self._num_track_dummy, self._num_axis, self._num_class))
                        else:
                            extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                        for f_row in extra_feat:
                            self._circ_buf_feat.append(f_row)
                        for l_row in extra_labels:
                            self._circ_buf_label.append(l_row)

                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_f_bins * self._nb_ch))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_f_bins))

                if self._multi_accdoa is True:
                    label = np.zeros(
                        (self._label_batch_seq_len, self._num_track_dummy, self._num_axis, self._num_class))
                    for j in range(self._label_batch_seq_len):
                        label[j, :, :, :] = self._circ_buf_label.popleft()
                else:
                    label = np.zeros((self._label_batch_seq_len, self._label_len))
                    for j in range(self._label_batch_seq_len):
                        label[j, :] = self._circ_buf_label.popleft()




                # Split to sequences
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                feat = np.transpose(feat, (0, 2, 1, 3))

                label = self._split_in_seqs(label, self._label_seq_len)
                if self._multi_accdoa is True:
                    pass
                else:
                    mask = label[:, :, :self._nb_classes]
                    mask = np.tile(mask, 3)
                    label = mask * label[:, :, self._nb_classes:]

                yield feat, label

    def _split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        elif len(data.shape) == 4:  # for multi-ACCDOA with ADPIT
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2], data.shape[3]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_data_sizes(self):
        # 特征形状：（批次大小，通道数，序列长度，频率bins）
        feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._nb_f_bins)
        if self._is_eval:
            label_shape = None
        else:
            if self._multi_accdoa is True:
                # 多轨迹标签形状：（批次大小，序列长度，类别数×3×3）
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes * 3 * 3)
            else:
                # 单轨迹标签形状：（批次大小，序列长度，类别数×3）
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes * 3)
        return feat_shape, label_shape

    def _get_filelist_n_frames(self,filename):
        self._filenames_list.append(filename)  # 将文件添加到列表
        temp_feat = np.load(os.path.join(self._feat_dir, filename))  # 加载特征
        # 累计总帧数（仅保留序列长度整数倍的部分）
        self.total_frames += (temp_feat.shape[0] - (temp_feat.shape[0] % self._feature_seq_len))
        # 更新最大帧数
        if temp_feat.shape[0] > self.max_frames:
            self.max_frames = temp_feat.shape[0]
        return temp_feat

    def get_nb_classes(self):
        return self._nb_classes

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._feat_cls.get_hop_len_sec()

    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._label_batch_seq_len

    def get_data_gen_mode(self):
        return self._is_eval

    def write_output_format_file(self, _out_file, _out_dict, is_eval=False):
        if is_eval:
            _out_dict = self.convert_output_format_cartesian_to_polar_eval(_out_dict)
        return self._feat_cls.write_output_format_file(_out_file, _out_dict, is_eval)

    def convert_output_format_cartesian_to_polar_eval(self, _out_dict):
        return self._feat_cls.convert_output_format_cartesian_to_polar_eval(_out_dict)
