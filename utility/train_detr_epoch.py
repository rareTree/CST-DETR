import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plot

plot.switch_backend('agg')


def train_detr_epoch(data_generator, optimizer, model, criterion, params, device, epoch_cnt):
    """
    1. 移除了循环内的 lr_sched 调用 (交给外部 ReduceLROnPlateau)。
    2. 支持返回分层学习率中的 'Head' 学习率。
    """
    nb_train_batches, train_loss = 0, 0.
    model.train()

    total_batches = data_generator.get_total_batches_in_data()

    # 使用 tqdm 显示进度条，保持和你原代码一致的体验
    with tqdm(total=total_batches) as pbar:
        for data, target in data_generator.generate():

            # 1. 数据准备
            # (保持和你原 train_epoch 一样的处理逻辑)
            data = torch.tensor(data).float().to(device)
            target = torch.tensor(target).float().to(device)

            optimizer.zero_grad()

            # 2. 前向传播
            # 使用 contiguous() 是个好习惯
            output = model(data.contiguous())

            # 3. 计算损失 (SetCriterion)
            loss = criterion(output, target)

            # 4. 反向传播
            loss.backward()

            # 5. 梯度裁剪 (这是 DETR 训练稳定的关键，原代码没有)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            train_loss += loss.item()
            nb_train_batches += 1

            if params['quick_test'] and nb_train_batches == 4:
                break

            pbar.update(1)

    train_loss /= nb_train_batches

    # 6. 获取当前学习率
    # 如果使用了分层学习率 (Backbone=Group0, Head=Group1)，我们返回 Head 的学习率以便观察
    if len(optimizer.param_groups) > 1:
        learning_rate = optimizer.param_groups[1]['lr']
    else:
        learning_rate = optimizer.param_groups[0]['lr']

    return train_loss, learning_rate