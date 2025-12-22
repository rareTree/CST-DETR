import torch
import torch.nn as nn
from einops import rearrange

class CST_attention(torch.nn.Module):
    def __init__(self, temp_embed_dim, params, kernel_size=None):
        super().__init__()
        self.nb_mel_bins = params['nb_mel_bins']  # 梅尔频谱的频率bin数量
        self.ChAtten_dca = params['ChAtten_DCA']  # 是否使用DCA（Divided Channel Attention）通道注意力
        self.ChAtten_ule = params['ChAtten_ULE']  # 是否使用ULE（Unfolded Local Embedding）通道注意力
        self.FreqAtten = params['FreqAtten']  # 是否使用频谱注意力
        self.linear_layer = params['LinearLayer']  # 是否使用线性层增强注意力输出
        self.dropout_rate = params['dropout_rate']
        self.temp_embed_dim = temp_embed_dim  # 时间维度的嵌入维度
        self.nb_ch = 7

        # Channel attention w. Divided Channel Attention (DCA) ---------------------------------------------#
        if self.ChAtten_dca:
            self.ch_attn_embed_dim = params['nb_cnn2d_filt']  # 通道注意力的嵌入维度（设为CNN输出特征数，如64）
            # 初始化多头自注意力（MHA）：输入维度、头数、dropout、batch_first=True表示输入格式为(batch, seq_len, embed_dim)
            self.ch_mhsa = nn.MultiheadAttention(embed_dim=self.ch_attn_embed_dim, num_heads=params['nb_heads'],
                                      dropout=self.dropout_rate, batch_first=True)
            self.ch_layer_norm = nn.LayerNorm(self.temp_embed_dim)  # 通道注意力后的层归一化
            if self.linear_layer:   # 若启用线性层，则添加线性变换（维度不变）
                self.ch_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        # Channel attention w. Unfolded Local Embedding (ULE) ----------------------------------------------#
        if self.ChAtten_ule:
            # --- 核心修改开始 ---
            # 如果传入了特定的 kernel_size (tuple), 就直接使用
            if kernel_size is not None:
                self.patch_size = kernel_size  # (T, F)
                self.patch_size_t, self.patch_size_f = self.patch_size
            else:
                # 否则使用旧的默认逻辑 (兼容性保留)
                self.patch_size_t = 25 if params['t_pooling_loc'] == 'end' else 10
                self.patch_size_f = 4
                self.patch_size = (self.patch_size_t, self.patch_size_f)
            # 计算频率维度和时间维度的大小（基于池化后的特征）
            self.freq_dim = int(self.nb_mel_bins / torch.prod(torch.Tensor(params['f_pool_size'])))
            self.temp_dim = 250 if params['t_pooling_loc']=='end' else 50
            # 初始化unfold（提取补丁）和fold（还原补丁）操作
            self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)  # 无重叠提取补丁
            self.fold = nn.Fold(output_size=(self.temp_dim, self.freq_dim), kernel_size=self.patch_size, stride=self.patch_size)
            self.ch_attn_embed_dim = int(self.patch_size_t * self.patch_size_f)  # 通道注意力嵌入维度=补丁大小
            # 初始化ULE模式的多头自注意力（头数根据时间池化位置调整）
            self.ch_mhsa = nn.MultiheadAttention(embed_dim=self.ch_attn_embed_dim, num_heads=10 if params['t_pooling_loc']=='end' else params['nb_heads'],
                                                 dropout=self.dropout_rate, batch_first=True)
            self.ch_layer_norm = nn.LayerNorm(self.temp_embed_dim)
            if self.linear_layer:
                self.ch_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        # Spectral attention -------------------------------------------------------------------------------#
        if self.FreqAtten:
            self.sp_attn_embed_dim = params['nb_cnn2d_filt']  # 64
            self.embed_dim_4_freq_attn = params['nb_cnn2d_filt'] # Update the temp embedding if freq attention is applied
            # 初始化频谱多头自注意力
            self.sp_mhsa = nn.MultiheadAttention(embed_dim=self.sp_attn_embed_dim, num_heads=params['nb_heads'],
                                      dropout=self.dropout_rate, batch_first=True)
            self.sp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
            if self.linear_layer:
                self.sp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        # temporal attention -----------------------------------------------------------------------------------#
        # 时间注意力的嵌入维度：若启用频谱注意力则使用其嵌入维度，否则使用默认嵌入维度
        self.temp_mhsa = nn.MultiheadAttention(embed_dim=self.embed_dim_4_freq_attn if params['FreqAtten'] else self.embed_dim,
                                  num_heads=params['nb_heads'],
                                  dropout=self.dropout_rate, batch_first=True)
        self.temp_layer_norm = nn.LayerNorm(self.temp_embed_dim)
        if self.linear_layer:
            self.temp_linear = nn.Linear(self.temp_embed_dim, self.temp_embed_dim)

        self.activation = nn.GELU()
        self.drop_out = nn.Dropout(self.dropout_rate if self.dropout_rate > 0. else nn.Identity())

        self.apply(self._init_weights)

        self.print_result = params['print_result']

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

    def forward(self,x, M, C, T, F):
        # 分支1：使用ULE通道注意力（CST-attention(ULE)）
        if self.ChAtten_ule: # CST-attention(ULE)
            if self.print_result:
                print("此处为使用ULE通道")
            # channel attention unfold
            B = x.size(0) # 获取批次大小
            x_init = x.clone() # 保存原始输入用于残差连接

            # 通道注意力：通过unfold提取局部补丁作为嵌入
            # 调整维度：(b, t, f*c) → (b, c, t, f)（c=通道数，t=时间，f=频率）
            if self.print_result:
                print(f"此处 x.shape is {x.shape}")
            x_unfold_in = rearrange(x_init, ' b t (f c) -> b c t f', c=C, t=T, f=F).contiguous()
            if self.print_result:
                print(f"经过 x_unfold_in = rearrange(x_init, ' b t (f c) -> b c t f', c=C, t=T, f=F).contiguous() 后，x.shape is {x_unfold_in.shape}")
            # 提取补丁：输出形状为(b, c*u, tf)，u=补丁元素数，tf=补丁总数
            x_unfold = self.unfold(x_unfold_in) # unfold for additional embedding for channel attention
            if self.print_result:
                print(f"经过 x_unfold = self.unfold(x_unfold_in) 后，x.shape is {x_unfold.shape}")
            x_unfold = rearrange(x_unfold, 'b (c u) tf -> (b tf) c u', c=C).contiguous()
            if self.print_result:
                print(f"经过 x_unfold = rearrange(x_unfold, 'b (c u) tf -> (b tf) c u', c=C).contiguous() 后，x.shape is {x_unfold.shape}")

            # 执行通道多头自注意力计算
                print("开始进行通道多头注意力 xc, _ = self.ch_mhsa(x_unfold, x_unfold, x_unfold)")
            xc, _ = self.ch_mhsa(x_unfold, x_unfold, x_unfold)
            if self.print_result:
                print(f"注意力输出维度 xc 为：{xc.shape}")

            # 还原维度：(b*tf, c, u) → (b, c*u, tf) → 折叠为(b, c, t, f)
            xc = rearrange(xc, '(b tf) c u -> b (c u) tf', b=B).contiguous()
            if self.print_result:
                print(f"此处还原维度，经过 xc = rearrange(xc, '(b tf) c u -> b (c u) tf', b=B).contiguous() 后,xc.shape is {xc.shape}")
            xc = self.fold(xc)  # fold to rearrange
            if self.print_result:
                print(f"经过 xc = self.fold(xc) 后,xc.shape is {xc.shape}")
            xc = rearrange(xc, 'b c t f -> b t (f c)').contiguous()
            if self.print_result:
                print(f"经过 xc = rearrange(xc, 'b c t f -> b t (f c)').contiguous() 后,xc.shape is {xc.shape}")

            # 残差连接+线性变换（可选）+dropout+层归一化
            if self.linear_layer:
                xc = self.activation(self.ch_linear(xc))
                if self.print_result:
                    print("此处进行 xc = self.activation(self.ch_linear(xc))")
            xc = xc + x_init
            if self.print_result:
                print("残差连接，xc = xc + x_init")
            if self.dropout_rate:
                xc = self.drop_out(xc)
                if self.print_result:
                    print("xc = self.drop_out(xc)")
            xc = self.ch_layer_norm(xc)
            if self.print_result:
                print("xc = self.ch_layer_norm(xc)")

            # spectral attention
                print(f"现在 xc.shape is {xc.shape}")
            xs = rearrange(xc, ' b t (f c) -> (b t) f c', f=F).contiguous()
            if self.print_result:
                print(f"经过 xs = rearrange(xc, ' b t (f c) -> (b t) f c', f=F).contiguous() 后， xs.shape is {xs.shape}")
                print("开始进行频谱多头注意力 xs, _ = self.sp_mhsa(xs, xs, xs)")
            xs, _ = self.sp_mhsa(xs, xs, xs)
            if self.print_result:
                print(f"注意力输出维度 xs 为：{xs.shape}")
            xs = rearrange(xs, ' (b t) f c -> b t (f c)', t=T).contiguous()
            if self.print_result:
                print(f"经过 xs = rearrange(xs, ' (b t) f c -> b t (f c)', t=T).contiguous() 后， xs.shape is {xs.shape}")
            if self.linear_layer:
                xs = self.activation(self.sp_linear(xs))
                if self.print_result:
                    print("此处进行 xs = self.activation(self.sp_linear(xs))")
            xs = xs + xc
            if self.print_result:
                print("残差连接，xs = xs + xc")
            if self.dropout_rate:
                xs = self.drop_out(xs)
                if self.print_result:
                    print("xs = self.drop_out(xs)")
            xs = self.sp_layer_norm(xs)
            if self.print_result:
                print("xs = self.sp_layer_norm(xs)")

            # temporal attention
                print(f"现在 xs.shape is {xs.shape}")
            xt = rearrange(xs, ' b t (f c) -> (b f) t c', f=F).contiguous()
            if self.print_result:
                print(f"经过 xt = rearrange(xs, ' b t (f c) -> (b f) t c', f=F).contiguous() 后， xt.shape is {xt.shape}")
                print("开始进行时间多头注意力 xt, _ = self.temp_mhsa(xt, xt, xt)")
            xt, _ = self.temp_mhsa(xt, xt, xt)
            if self.print_result:
                print(f"注意力输出维度 xt 为：{xt.shape}")
            xt = rearrange(xt, ' (b f) t c -> b t (f c)', f=F).contiguous()
            if self.print_result:
                print(f"经过 xt = rearrange(xt, ' (b f) t c -> b t (f c)', f=F).contiguous() 后， xt.shape is {xt.shape}")
            if self.linear_layer:
                xt = self.activation(self.temp_linear(xt))
                if self.print_result:
                    print("此处进行 xt = self.activation(self.temp_linear(xt))")
            xt = xt + xs
            if self.print_result:
                print("残差连接，xt = xt + xs")
            if self.dropout_rate:
                xt = self.drop_out(xt)
                if self.print_result:
                    print("xt = self.drop_out(xt)")
            x = self.temp_layer_norm(xt)
            if self.print_result:
                print("x = self.temp_layer_norm(xt)")

        # 分支2：使用DCA通道注意力（CST-attention(DCA)）
        elif self.ChAtten_dca: #CST-attention (DCA)
            # channel attention
            x_init = x.clone()
            # 调整维度：(b, t, m*f*c) → (b*t*f, m, c)（m=麦克风通道数，将时间和频率合并到批次）
            xc = rearrange(x_init, 'b t (m f c)-> (b t f) m c', c=C, f=F).contiguous()

            xc, _ = self.ch_mhsa(xc, xc, xc)
            xc = rearrange(xc, ' (b t f) m c -> b t (f m c)', t=T, f=F).contiguous()
            if self.linear_layer:
                xc = self.activation(self.ch_linear(xc))
            xc = xc + x_init
            if self.dropout_rate:
                xc = self.drop_out(xc)
            xc = self.ch_layer_norm(xc)

            # spectral attention
            xs = rearrange(xc, ' b t (f m c) -> (b t m) f c', c=C, t=T, f=F).contiguous()
            xs, _ = self.sp_mhsa(xs, xs, xs)
            xs = rearrange(xs, ' (b t m) f c -> b t (f m c)', m=M, t=T).contiguous()
            if self.linear_layer:
                xs = self.activation(self.sp_linear(xs))
            xs = xs + xc
            if self.dropout_rate:
                xs = self.drop_out(xs)
            xs = self.sp_layer_norm(xs)

            # temporal attention
            xt = rearrange(xs, ' b t (f m c) -> (b f m) t c', m=M, f=F).contiguous()
            xt, _ = self.temp_mhsa(xt, xt, xt)
            xt = rearrange(xt, ' (b f m) t c -> b t (f m c)', m=M, f=F).contiguous()
            if self.linear_layer:
                xt = self.activation(self.temp_linear(xt))
            xt = xt + xs
            if self.dropout_rate:
                xt = self.drop_out(xt)
            x = self.temp_layer_norm(xt)

        # 分支3：仅使用频谱+时间注意力（DST-attention）
        elif self.FreqAtten: # DST-attention
            x_init = x.clone()
            # spectral attention
            x_attn_in = rearrange(x_init, ' b t (f c) -> (b t) f c', f=F).contiguous()
            xs, _ = self.sp_mhsa(x_attn_in, x_attn_in, x_attn_in)
            xs = rearrange(xs, ' (b t) f c -> b t (f c)', t=T).contiguous()
            if self.linear_layer:
                xs = self.activation(self.sp_linear(xs))
            xs = xs + x_init
            if self.dropout_rate:
                xs = self.drop_out(xs)
            xs = self.sp_layer_norm(xs)

            # temporal attention
            xt = rearrange(xs, ' b t (f c) -> (b f) t c', c=C).contiguous()
            xt, _ = self.temp_mhsa(xt, xt, xt)
            xt = rearrange(xt, ' (b f) t c -> b t (f c)', f=F).contiguous()
            if self.linear_layer:
                xt = self.activation(self.temp_linear(xt))
            xt = xt + xs
            if self.dropout_rate:
                xt = self.drop_out(xt)
            x = self.temp_layer_norm(xt)

        # 分支4：仅使用基本时间注意力
        else: # Basic Temporal Attention
            x_attn_in = x
            x, _ = self.temp_mhsa(x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.temp_layer_norm(x)

        return x

class CST_encoder(torch.nn.Module):
    def __init__(self, temp_embed_dim, params):
        super().__init__()
        self.freq_atten = params['FreqAtten']
        self.ch_atten_dca = params['ChAtten_DCA']
        self.ch_atten_ule = params['ChAtten_ULE']
        self.nb_ch = 7
        n_layers = params['nb_self_attn_layers']

        msule_kernels = [(25, 4), (10, 4), (5, 4), (5, 2)]
        for i in range(n_layers):
            # 获取当前层的 kernel_size，如果层数超过列表长度，则沿用最后一个
            current_kernel = msule_kernels[i] if i < len(msule_kernels) else msule_kernels[-1]

            # 打印一下配置，让你放心
            print(f"CST Layer {i + 1}/{n_layers}: MSULE Kernel set to {current_kernel}")

            self.block_list.append(CST_attention(
                temp_embed_dim=temp_embed_dim,
                params=params,
                kernel_size=current_kernel  # 传入动态核
            ))

    def forward(self, x):
        B, C, T, F = x.size()
        M = self.nb_ch # Number of Microphone Channels

        # CST-attention
        # 若使用DCA通道注意力，调整输入维度以融入麦克风通道信息
        if self.ch_atten_dca:
            B = B // M  # Real Batch
            x = rearrange(x, '(b m) c t f -> b t (m f c)', b=B, m=M).contiguous()

        # DST-attention
        # 若使用ULE通道注意力或频谱注意力，调整输入维度为时间×(频率×特征通道)
        if self.ch_atten_ule or self.freq_atten:
            x = rearrange(x, 'b c t f -> b t (f c)').contiguous()

        for block in self.block_list:
            x = block(x, M, C, T, F)

        return x
