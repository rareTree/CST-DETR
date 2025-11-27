import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# einops.rearrange用于灵活调整张量维度（类似reshape，但更简洁）

# 定义3x3卷积层的快捷函数（常用于ResNet等结构）
# in_planes：输入通道数；out_planes：输出通道数；stride：步长（默认1）
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# 将输入转换为元组（若输入不是元组），方便处理卷积的kernel_size/stride等参数（可能需要分别指定高/宽）
def make_pairs(x):
    """make the int -> tuple
    """
    return x if isinstance(x, tuple) else (x, x)

# 生成相对距离矩阵（用于注意力机制中的相对位置编码）
def generate_relative_distance(number_size):
    """return relative distance, (number_size**2, number_size**2, 2)
    """
    """
        生成二维网格中所有位置对的相对距离
        参数：number_size：网格的边长（如8则生成8x8网格）
        返回：(number_size², number_size², 2)的张量，每个元素是两个位置的(x,y)相对偏移
    """
    # 生成网格中所有位置的坐标：[[0,0], [0,1], ..., [number_size-1, number_size-1]]
    indices = torch.tensor(np.array([[x, y] for x in range(number_size) for y in range(number_size)]))
    # 计算所有位置对的相对距离：indices[None,:,:]是(1, N², 2)，indices[:,None,:]是(N², 1, 2)，相减后得到(N², N², 2)
    distances = indices[None, :, :] - indices[:, None, :]
    distances = distances + number_size - 1   # shift the zeros postion
    # 偏移距离（将负数转为非负数，方便后续作为索引）
    return distances


#####################################################################################################################
### Layers for CST-former
#####################################################################################################################
class GRU_layer(torch.nn.Module):
    """
    GRU layer for baseline
    """
    def __init__(self, in_feat_shape, params):
        super().__init__()
        if params["baseline"]:
            # 计算GRU输入维度：CNN输出的特征通道数 × 特征图宽度的1/4（可能是经过下采样后的尺寸）
            self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / 4))
            # 定义GRU层：输入尺寸=gru_input_dim，隐藏层尺寸=params['rnn_size']，层数=params['nb_rnn_layers']
            # batch_first=True表示输入格式为(batch, seq_len, input_size)
            # dropout=params['dropout_rate']表示层间dropout，bidirectional=True表示双向GRU
            self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)

    def forward(self,x):
        # 调整维度：将时序维度放到第二维（假设输入是(batch, channels, seq_len, ...)）
        x = x.transpose(1, 2).contiguous()
        # 展平特征维度：将除batch和seq_len外的维度合并为特征向量
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        # GRU前向传播：返回输出和最终隐藏状态（这里忽略隐藏状态）
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        # 双向GRU的门控操作：将前向和后向输出相乘（类似门控机制，增强有效特征）
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]
        return x

class FC_layer(torch.nn.Module):
    """
    Fully Connected layer for baseline

    Args:
        out_shape (int): output shape for SLED
                         ex. 39 for single-ACCDOA, 117 for multi-ACCDOA
        temp_embed_dim (int): the input size
        params : parameters from parameter.py
    """
    def __init__(self, out_shape,temp_embed_dim, params):
        super().__init__()

        self.fnn_list = torch.nn.ModuleList()  # 用ModuleList存储全连接层（方便管理）
        # 根据参数中的全连接层数，添加中间层
        if params['nb_fnn_layers']:
            # 第一层输入是temp_embed_dim，后续层输入是params['fnn_size']，输出都是params['fnn_size']
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    nn.Linear(params['fnn_size'] if fc_cnt else temp_embed_dim, params['fnn_size'], bias=True))
        self.fnn_list.append(
            nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else temp_embed_dim, out_shape[-1],
                      bias=True))

    def forward(self, x:torch.Tensor):
        print("这里是全连接层")
        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = torch.tanh(self.fnn_list[-1](x))
        return doa


#####################################################################################################################
### Convolution meets transformer (CMT)
#####################################################################################################################

class LocalPerceptionUint(torch.nn.Module):
    """局部感知单元（用于捕获局部空间特征，类似卷积的作用）"""
    def __init__(self, dim, act=False):
        super(LocalPerceptionUint, self).__init__()
        self.act = act  # 是否使用激活函数和批归一化
        self.conv_3x3_dw = ConvDW3x3(dim)  # 3x3深度卷积（逐通道卷积）
        if self.act:
            self.actation = nn.Sequential(
                nn.GELU(),  # 激活函数
                nn.BatchNorm2d(dim)  # 批归一化（稳定训练）
            )

        self.initialize_weights()
        self.print_result = False

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.print_result:
            print("This is LPU")
        if self.act:  # 若需要激活，先卷积再激活+批归一化
            if self.print_result:
                print(f"input size is {x.shape}")
                print("经过 self.actation(self.conv_3x3_dw(x))")
            out = self.actation(self.conv_3x3_dw(x))
            if self.print_result:
                print(f"现在 x.shape is {out.shape}")
            return out
        else:  # 否则直接返回卷积结果
            if self.print_result:
                print(f"input size is {x.shape}")
            out = self.conv_3x3_dw(x)
            if self.print_result:
                print(f"经过 out = self.conv_3x3_dw(x) 后，x.shape is {out.shape}")
            return out

class InvertedResidualFeedForward(torch.nn.Module):
    """倒置残差前馈网络（类似MobileNet的倒置残差结构，用于特征转换）"""
    def __init__(self, dim, dim_ratio=4.):
        super(InvertedResidualFeedForward, self).__init__()
        output_dim = int(dim_ratio * dim)  # 中间层维度（输入dim的dim_ratio倍，默认4倍）
        # 1. 1x1卷积+GELU+批归一化（升维+非线性激活）
        self.conv1x1_gelu_bn = ConvGeluBN(
            in_channel=dim,
            out_channel=output_dim,
            kernel_size=1,  # 1x1卷积（仅调整通道数，不改变尺寸）
            stride_size=1,
            padding=0
        )

        # 2. 3x3深度卷积（逐通道局部特征细化）
        self.conv3x3_dw = ConvDW3x3(dim=output_dim)
        # 3. 激活+批归一化（增强非线性+稳定分布）
        self.act = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm2d(output_dim)
        )
        # 4. 1x1卷积+批归一化（降维+还原通道数）
        self.conv1x1_pw = nn.Sequential(
            nn.Conv2d(output_dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim)
        )

        self.initialize_weights()
        self.print_result = False

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层用Kaiming均匀分布（适配ReLU/GELU激活）
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # BN层：偏置=0，权重=1（保持归一化后分布稳定）
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.print_result:
            print("This is InvertedResidualFeedForward")
            print(f"input x.size is {x.shape}")
        # 步骤1：1x1卷积升维+GELU+BN
        x = self.conv1x1_gelu_bn(x)
        if self.print_result:
            print(f"步骤1：1x1卷积升维+GELU+BN,x = self.conv1x1_gelu_bn(x),x.shape is {x.shape}")
        # 步骤2：残差连接（原升维特征 + 深度卷积+激活+BN后的特征）
        out = x + self.act(self.conv3x3_dw(x))
        if self.print_result:
            print(f"步骤2：残差连接（原升维特征 + 深度卷积+激活+BN后的特征）,out = x + self.act(self.conv3x3_dw(x)),x.shape is {out.shape}")
        # 步骤3：1x1卷积降维+BN
        out = self.conv1x1_pw(out)
        if self.print_result:
            print(f"步骤3：1x1卷积降维+BN,out = self.conv1x1_pw(out),x.shape is {out.shape}")
        return out


class ConvDW3x3(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(ConvDW3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=make_pairs(kernel_size),  # 卷积核尺寸（默认3x3）
            padding=make_pairs(1),
            groups=dim)  # 分组卷积参数（=dim，深度卷积的核心）

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvGeluBN(nn.Module):
    """卷积+GELU激活+批归一化的组合层（用于快速构建特征转换模块）"""
    def __init__(self, in_channel, out_channel, kernel_size, stride_size, padding=1):
        """build the conv3x3 + gelu + bn module
        """
        super(ConvGeluBN, self).__init__()
        # 将参数转为元组（支持不同的高/宽设置，如kernel_size=(3,5)
        self.kernel_size = make_pairs(kernel_size)
        self.stride_size = make_pairs(stride_size)
        self.padding_size = make_pairs(padding)
        self.in_channel = in_channel
        self.out_channel = out_channel
        # 核心组合：卷积 → GELU → 批归一化
        self.conv3x3_gelu_bn = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=self.out_channel,
                      kernel_size=self.kernel_size,
                      stride=self.stride_size,
                      padding=self.padding_size),
            nn.GELU(),
            nn.BatchNorm2d(self.out_channel)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.conv3x3_gelu_bn(x)
        return x


#####################################################################################################################
### Convolutional Blocks
#####################################################################################################################

class ConvBlock(nn.Module):
    """单卷积块（卷积+批归一化+ReLU，基础特征提取）"""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)  # 批归一化（输出通道数为out_channels）

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class ConvBlockTwo(nn.Module):
    """双卷积块（两个ConvBlock串联，增强特征提取能力）"""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


#####################################################################################################################
### ResNet
#####################################################################################################################

class ResidualBlock(nn.Module):
    """残差块（双卷积+残差连接，适配深层网络）"""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        # First Layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_init):
        identity = x_init.clone()  # 保存输入作为残差（跳跃连接）
        x = F.relu(self.bn1(self.conv1(x_init)))
        x = F.relu(self.bn2(self.conv2(x)) + identity)
        return x


#####################################################################################################################
### Squeeze and Excitation  挤压和激励
#####################################################################################################################

class SELayer(nn.Module):
    """基础SE层（通过挤压和激励机制增强重要通道特征）"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 全局平均池化：将每个通道的特征图压缩为1x1（挤压操作）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层（激励操作）：先降维（channel→channel//reduction），再升维（channel//reduction→channel），最后sigmoid输出权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),  # 激活
            nn.Linear(channel // reduction, channel, bias=False),  # 升维
            nn.Sigmoid()  # 输出通道权重（0-1）
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入形状：(batch, channel, height, width)
        # 挤压：全局平均池化→展平为(batch, channel)
        y = self.avg_pool(x).view(b, c)
        # 激励：通过全连接层生成权重→reshape为(batch, channel, 1, 1)
        y = self.fc(y).view(b, c, 1, 1)
        # 应用注意力：每个通道特征 × 对应权重
        return x * y.expand_as(x)


class SE_MSCAM(nn.Module):
    """改进的SE层（结合多尺度信息的通道注意力）  Multi-Scale Channel Attention Module"""
    def __init__(self, channel, reduction=16):
        super(SE_MSCAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化（挤压）
        # 第一个SE分支：基于全局池化的特征（全局尺度）
        self.se1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        # 第二个SE分支：基于原始特征图的特征（局部+全局尺度）
        self.se2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.activation = nn.Sigmoid()  # 输出权重（0-1）

    def forward(self, x):
        b, c, _, _ = x.size()
        # 分支1：全局池化→1x1卷积处理（捕捉全局通道依赖）
        y1 = self.avg_pool(x).view(b, c, 1, 1)
        y1 = self.se1(y1).view(b, c, 1, 1)

        # 分支2：直接对原始特征图做1x1卷积处理（捕捉局部+全局通道依赖）
        y2 = self.se2(x)

        # 融合两个分支的结果→激活生成权重
        y = y1.expand_as(y2) + y2
        y = self.activation(y)

        # 应用注意力
        return x * y


class SEBasicBlock(nn.Module):
    """带SE注意力的残差块（将SE机制融入ResNet基础块）"""
    expansion = 1  # 残差块输出通道数的扩展倍数（1表示不扩展）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 *, reduction=16, MSCAM=False):
        super(SEBasicBlock, self).__init__()
        # 第一层3x3卷积
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批归一化
        self.relu = nn.ReLU(inplace=True)   # ReLU激活
        # 第二层3x3卷积（步长1，保持尺寸）
        self.conv2 = conv3x3(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 根据是否使用MSCAM选择SE层类型
        if not MSCAM:
            self.se = SELayer(out_channels, reduction)
        else:
            self.se = SE_MSCAM(out_channels, reduction)

        # 下采样模块（当输入输出通道/尺寸不同时使用）
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # 残差（跳跃连接）
        # 第一层：卷积→批归一化→ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层：卷积→批归一化→SE注意力
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # 应用通道注意力

        # 若需要下采样（输入输出尺寸/通道不同），对残差进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接：注意力处理后的特征 + 下采样后的残差→ReLU
        out += residual
        out = self.relu(out)

        return out

#####################################################################################################################
### Attention Layers for CMT Split (To apply LPU&IRFFN on each attention layers)
#####################################################################################################################
class Spec_attention(torch.nn.Module):
    """频谱注意力层（捕获频谱维度的长距离依赖）"""
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        # self.params = params
        self.dropout_rate = params['dropout_rate']  # dropout比率
        self.linear_layer = params['linear_layer']  # 是否使用线性层
        self.temp_embed_dim = temp_embed_dim  # 时间嵌入维度

        # Spectral attention -------------------------------------------------------------------------------#
        # 频谱注意力配置
        self.sp_attn_embed_dim = params['nb_cnn2d_filt']  # 频谱注意力的嵌入维度（64）
        # 多头自注意力（Multi-Head Self-Attention）：输入维度=sp_attn_embed_dim，头数=params['nb_heads']
        self.sp_mhsa = nn.MultiheadAttention(embed_dim=self.sp_attn_embed_dim, num_heads=params['nb_heads'],
                                  dropout=params['dropout_rate'], batch_first=True)
        self.sp_layer_norm = nn.LayerNorm(self.temp_embed_dim)  # 层归一化（输入维度=temp_embed_dim）
        # 若启用线性层，定义频谱注意力后的线性变换
        if self.params['LinearLayer']:
            self.sp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        self.activation = nn.GELU()  # GELU激活函数
        # 定义dropout层（若dropout_rate>0则使用，否则用恒等映射）
        self.drop_out = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0. else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x, C, T, F):
        # x：输入特征；C：通道数；T：时间步长；F：频率维度
        # spectral attention
        x_init = x  # 保存输入用于残差连接
        # 调整维度：将输入从(batch, time, (freq*channel))重排为(batch*time, freq, channel)
        # 目的：将时间维度合并到batch，以频率为序列长度，便于计算频谱注意力
        x_attn_in = rearrange(x_init, ' b t (f c) -> (b t) f c', c=C,f=F).contiguous()
        # 频谱多头自注意力：输入=输出（(b*t, f, c)），返回注意力输出和注意力权重（忽略权重）
        xs, _ = self.sp_mhsa(x_attn_in, x_attn_in, x_attn_in)
        # 维度重排：将注意力输出从(batch*time, freq, channel)转回(batch, time, (freq*channel))
        xs = rearrange(xs, ' (b t) f c -> b t (f c)', t=T).contiguous()
        # 若启用线性层，应用线性变换+GELU
        if self.linear_layer:
            xs = self.activation(self.sp_linear(xs))
        # 残差连接：注意力输出 + 原始输入
        xs = xs + x_init
        if self.dropout_rate:
            xs = self.drop_out(xs)
        # 层归一化
        x_out = self.sp_layer_norm(xs)
        return x_out

class Temp_attention(torch.nn.Module):
    """时间注意力层（捕获时间维度的长距离依赖）"""
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        # self.params = params
        self.dropout_rate = params['dropout_rate']
        self.linear_layer = params['linear_layer']
        self.temp_embed_dim = temp_embed_dim  # 时间嵌入维度（=F×C）
        # 时间注意力的嵌入维度（根据是否启用频率注意力动态调整）
        self.embed_dim_4_freq_attn = params['nb_cnn2d_filt']  # Update the temp embedding if freq attention is applied
        # temporal attention -----------------------------------------------------------------------------------#
        # 多头自注意力：输入维度由FreqAtten参数决定
        self.temp_mhsa = nn.MultiheadAttention(embed_dim=self.embed_dim_4_freq_attn if params['FreqAtten'] else self.temp_embed_dim,
                                  num_heads=params['nb_heads'],
                                  dropout=params['dropout_rate'], batch_first=True)
        self.temp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
        if self.params['LinearLayer']:
            self.temp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        self.activation = nn.GELU()
        self.drop_out = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0. else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x, C, T, F):
        # x：输入特征；C：通道数；T：时间步长；F：频率维度
        # temporal attention
        x_init = x
        xt = rearrange(x_init, ' b t (f c) -> (b f) t c', c=C).contiguous()
        xt, _ = self.temp_mhsa(xt, xt, xt)
        xt = rearrange(xt, ' (b f) t c -> b t (f c)', f=F).contiguous()
        if self.linear_layer:
            xt = self.activation(self.temp_linear(xt))
        xt = xt + x_init
        if self.dropout_rate:
            xt = self.drop_out(xt)
        x_out = self.temp_layer_norm(xt)
        return x_out