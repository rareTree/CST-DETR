import os
import sys
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plot
from cls import cls_feature_class as cls_feature_class, cls_data_generator as cls_data_generator
import parameters as parameters
import time
import json
from time import gmtime, strftime
import torch
import torch.optim as optim
from utility.graphs import draw_loss

plot.switch_backend('agg')
from cls.cls_compute_seld_results import ComputeSELDResults
from utility.load_state_dict import load_state_dict
from architecture import CST_former_model as model_architecture
from utility.test_epoch import test_epoch
from utility.test_detr_epoch import test_detr_epoch
from utility.train_epoch import train_epoch as train_epoch
from utility.train_detr_epoch import train_detr_epoch
from utility.loss_adpit import MSELoss_ADPIT
from architecture.DETR_details.DCST_loss import HungarianMatcher, SetCriterion
from utility.lr_sched import adjust_learning_rate


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.
    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1
    """
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inpluts for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    # os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
    device = torch.device('cuda')

    # ---------------------------------------------- (For Reproducibility)
    # fix the seed for reproducibility
    seed = 2025
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    # ----------------------------------------------
    torch.autograd.set_detect_anomaly(True)

    # use parameter set defined by user
    task_id = '1000' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    # Training setup
    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'dev':
        if '2020' in params['dataset_dir']:
            test_splits = [1]
            val_splits = [2]
            train_splits = [[3, 4, 5, 6]]

        elif '2021' in params['dataset_dir']:
            test_splits = [6]
            val_splits = [5]
            train_splits = [[1, 2, 3, 4]]

        elif '2022' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]]
        elif '2023' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]]
        else:
            print('ERROR: Unknown dataset splits')
            exit()

    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print(
            '------------------------------------      SPLIT {}   -----------------------------------------------'.format(
                split))
        print('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        loc_feat = params['dataset']
        if params['dataset'] == 'mic':
            if params['use_salsalite']:
                loc_feat = '{}_salsa'.format(params['dataset'])
            else:
                loc_feat = '{}_gcc'.format(params['dataset'])
        loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

        # ----------------------------------------------
        unique_name = '{}_{}_{}_split{}_{}_{}'.format(
            task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat
        )
        cls_feature_class.create_folder(os.path.join(params['save_dir'], unique_name, params['model_dir']))
        model_name = os.path.join(params['save_dir'], unique_name, params['model_dir'], 'model.h5')
        latest_model_name = os.path.join(params['save_dir'], unique_name, params['model_dir'], 'model_latest.h5')
        print("unique_name: {}\n".format(unique_name))
        # ----------------------------------------------

        # Load train and validation data
        print('Loading training dataset:')
        data_gen_train = cls_data_generator.DataGenerator(
            params=params, split=train_splits[split_cnt]
        )

        print('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False, per_file=True
        )

        # Collect i/o data size and load model configuration
        data_in, data_out = data_gen_train.get_data_sizes()
        model = model_architecture.CST_former(data_in, data_out, params)
        matcher = HungarianMatcher(cost_class=4.0, cost_doa=1.0)
        weight_dict = {'loss_class': 2.0, 'loss_doa': 2.0}

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)

        if params['finetune_mode']:
            print('Running in finetuning mode.')
            model = load_state_dict(model, params['pretrained_model_weights'])
        print('---------------- SELD-net -------------------')
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))
        print(
            'MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n, rnn_size: {}\n, nb_attention_blocks: {}\n, fnn_size: {}\n'.format(
                params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'],
                params['rnn_size'], params['nb_self_attn_layers'],
                params['fnn_size']))
        print(model)

        # Dump results in DCASE output format for calculating final scores
        # ----------------------------------------------
        # 生成带时间戳的DCASE格式输出文件夹
        dcase_output_folder = os.path.join(params["save_dir"], unique_name, params['dcase_output_dir'],
                                           strftime("%Y%m%d%H%M%S", gmtime()))
        # ----------------------------------------------
        # 验证集结果输出文件夹
        dcase_output_val_folder = os.path.join(dcase_output_folder, 'val')
        # 创建文件夹（若存在则先删除）
        cls_feature_class.delete_and_create_folder(dcase_output_val_folder)
        print('Dumping recording-wise val results in: {}'.format(dcase_output_val_folder))

        # Initialize evaluation metric class
        score_obj = ComputeSELDResults(params)  # 初始化SELD指标计算对象（用于计算ER、F值、定位误差等）

        # start training
        best_val_epoch = -1
        best_ER, best_F, best_LE, best_LR, best_seld_scr = 1., 0., 180., 0., 9999
        patience_cnt = 0  # 早停计数器（记录连续未提升的epoch数）

        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        if params['lr'] is None:  # only base_lr is specified  # 若未指定学习率，根据batch_size和基础学习率计算
            params['lr'] = params['blr'] * params['batch_size'] / 256

        if params['use_detr']:
            print(">>> Initialize Optimizer with Layer-wise Learning Rate for DETR")

            base_lr = params['lr']
            backbone_lr = base_lr   # Backbone 降速 (例如 1e-4)
            head_lr = base_lr  # Head 全速 (例如 1e-3)

            # 2. 定义 Backbone 的关键词 (CST-former 原有部分)
            # 只要参数名包含这些词，就认为是 Backbone
            detr_keywords = ["query_generator", "decoder", "ffn", "positional_encoding", "enc_score_head",
                             "enc_doa_head", "ref_point_head", "pos_trans_norm"]

            # 3. 筛选参数
            backbone_params = []
            head_params = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue  # 跳过冻结的参数

                # 检查参数名是否包含 backbone 关键词
                if any(k in name for k in detr_keywords):
                    head_params.append(param)
                    # print(f"  [Backbone] {name}") # 调试时可取消注释
                else:
                    backbone_params.append(param)
                    # print(f"  [Head]     {name}") # 调试时可取消注释

            # 4. 构建参数组
            param_groups = [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": head_lr}
            ]

            optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
            print(f"    Backbone LR: {backbone_lr} (Params: {len(backbone_params)})")
            print(f"    Head LR:     {head_lr} (Params: {len(head_params)})")

            # 3. 初始化自动调度器 (ReduceLROnPlateau)
            # 当验证集分数不再下降时，自动降低学习率
            print(">>> Initialize ReduceLROnPlateau Scheduler")
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,  # 每次降速一半 (例如 0.001 -> 0.0005)
                patience=20,  # 忍耐 20 个 epoch
                verbose=True,
                min_lr=1e-6
            )

        else:
            # === 原模型逻辑 (保持不变) ===
            # 若未指定学习率，根据batch_size和基础学习率计算
            if params['lr'] is None:
                params['lr'] = params['blr'] * params['batch_size'] / 256
            optimizer = optim.Adam(model.parameters(), lr=params['lr'])

        # loss preparation
        if params['multi_accdoa'] is True:
            if params['use_detr'] is True:
                criterion = SetCriterion(
                    num_classes=13,
                    matcher=matcher,
                    weight_dict=weight_dict,
                    losses=['loss_class', 'loss_doa']
                ).to(device)
            else:
                criterion = MSELoss_ADPIT()  # 使用自定义的MSELoss_ADPIT损失函数

        else:
            criterion = nn.MSELoss()  # 否则使用标准MSE损失

        # Start Train_Valid Loss recording
        train_loss_rec = np.empty([params["nb_epochs"]])  # 存储训练损失的数组
        valid_loss_rec = np.empty([params["nb_epochs"]])  # 存储验证损失的数组
        valid_seld_scr_rec = np.empty([params["nb_epochs"]])  # 存储验证SELD分数的数组
        valid_ER_rec = np.empty([params["nb_epochs"]])  # 存储验证ER的数组
        valid_F_rec = np.empty([params["nb_epochs"]])  # 存储验证F值的数组
        valid_LE_rec = np.empty([params["nb_epochs"]])  # 存储验证LE的数组
        valid_LR_rec = np.empty([params["nb_epochs"]])  # 存储验证LR的数组
        if params['use_detr']:
            learning_rate_rec = np.zeros([params["nb_epochs"], 2])  # 建议改用 zeros
        else:
            learning_rate_rec = np.zeros([params["nb_epochs"]])

        for epoch_cnt in range(nb_epoch):  # 遍历每个训练epoch
            if params['lr_scheduler'] and epoch_cnt <= params['warmup_epochs']:
                warmup_ratio = epoch_cnt / float(params['warmup_epochs'])
                if warmup_ratio < 1e-6: warmup_ratio = 1e-6  # 防止除0

                if params['use_detr'] and len(optimizer.param_groups) > 1:
                    # 如果是 DETR 分层模式，分别缩放
                    # group[0] 是 Backbone, group[1] 是 Head
                    # 注意：这里需要用到我们在循环外定义的 backbone_lr 和 head_lr 变量
                    # 如果它们未定义（比如非DETR模式），代码会报错，所以加了上面的判断

                    # 重新计算当前轮次的目标 LR
                    cur_backbone_lr = backbone_lr * warmup_ratio
                    cur_head_lr = head_lr * warmup_ratio

                    optimizer.param_groups[0]['lr'] = cur_backbone_lr
                    optimizer.param_groups[1]['lr'] = cur_head_lr

                    print(
                        f"  [Warmup] Epoch {epoch_cnt}: Backbone LR set to {cur_backbone_lr:.8f}, Head LR set to {cur_head_lr:.8f}")

                else:
                    # 原有逻辑 (非 DETR 或 单一 LR)
                    current_lr = adjust_learning_rate(optimizer, epoch_cnt, params)
                    print(f"  [Warmup] Epoch {epoch_cnt}: LR set to {current_lr:.8f}")


            # ---------------------------------------------------------------------
            # TRAINING
            # ---------------------------------------------------------------------
            start_time = time.time()
            if params['use_detr']:
                train_loss, learning_rate = train_detr_epoch(data_gen_train, optimizer, model, criterion,
                                                             params, device, epoch_cnt)
            else:
                train_loss, learning_rate, = train_epoch(data_gen_train, optimizer, model, criterion,
                                                         params, device, epoch_cnt)  # 执行训练epoch，返回训练损失和当前学习率

            train_time = time.time() - start_time

            # ---------------------------------------------------------------------
            # VALIDATION
            # ---------------------------------------------------------------------
            start_time = time.time()
            if params['use_detr'] is True:
                val_loss = test_detr_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)
            else:
                val_loss = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)
            # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores
            val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(
                dcase_output_val_folder)


            if len(optimizer.param_groups) > 1:
                # 分层学习率：显示 "Backbone / Head"
                lr_backbone = optimizer.param_groups[0]['lr']
                lr_head = optimizer.param_groups[1]['lr']
                # 格式：CST学习率 / DETR学习率
                lr_str = "{:0.8f}/{:0.8f}".format(lr_backbone, lr_head)
                # 更新 learning_rate 变量为 Head 的 LR (用于画图记录等，保持主要指标)
            else:
                # 单一学习率
                learning_rate = optimizer.param_groups[0]['lr']
                lr_str = "{:0.5f}".format(learning_rate)

            val_time = time.time() - start_time

            # Save model if loss is good
            if val_seld_scr <= best_seld_scr:
                best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr
                torch.save(model.state_dict(), model_name)
                patience_cnt = 0
            else:
                patience_cnt += 1


            torch.save(model.state_dict(), latest_model_name)

            # Print stats
            print(
                'epoch: {}, time: {:0.2f}/{:0.2f}, '
                'lr:{},'
                'train_loss: {:0.4f}, val_loss: {:0.4f}, '
                'ER/F/LE/LR/SELD: {}, '
                'best_val_epoch: {} {}'.format(
                    epoch_cnt, train_time, val_time,
                    lr_str,
                    train_loss, val_loss,
                    '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr),
                    best_val_epoch,
                    '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_ER, best_F, best_LE, best_LR,
                                                                       best_seld_scr))
            )

            log_stats = {'epoch': epoch_cnt,
                         'lr': lr_str,
                         'train_loss': train_loss,
                         'valid_loss': val_loss,
                         'val_ER': val_ER, 'val_F': val_F, 'val_LE': val_LE, 'val_LR': val_LR,
                         'val_seld_scr': val_seld_scr, }

            with open(os.path.join(dcase_output_folder, "log_finetune.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            if params['use_detr'] and len(optimizer.param_groups) > 1:
                # 传入列表 [backbone_lr, head_lr]
                current_lr_to_draw = [optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']]
            else:
                current_lr_to_draw = learning_rate

            train_loss_rec, valid_loss_rec, valid_seld_scr_rec, valid_ER_rec, valid_F_rec, valid_LE_rec, valid_LR_rec, learning_rate_rec \
                = draw_loss(dcase_output_folder, epoch_cnt, best_val_epoch, current_lr_to_draw,
                            train_loss, val_loss, val_seld_scr, val_ER, val_F, val_LE, val_LR,
                            train_loss_rec, valid_loss_rec, valid_seld_scr_rec,
                            valid_ER_rec, valid_F_rec, valid_LE_rec, valid_LR_rec, learning_rate_rec)

            if params['use_detr'] and scheduler is not None:
                if epoch_cnt >= params['warmup_epochs']:
                    scheduler.step(val_seld_scr)  # 正常调度
                else:
                    # Warmup 期间，不让 Scheduler 乱动 (或者 step 但忽略结果)
                    pass

            if patience_cnt > params['patience']:
                break

        # ---------------------------------------------------------------------
        # Evaluate on unseen test data
        # ---------------------------------------------------------------------
        print('Load best model weights')
        # model.load_state_dict(torch.load(model_name, map_location='cpu'))

        print('Loading unseen test dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
        )

        # Dump results in DCASE output format for calculating final scores
        dcase_output_test_folder = os.path.join(dcase_output_folder, 'test')
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))

        _ = test_detr_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)
        use_jackknife = True
        test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(
            dcase_output_test_folder, is_jackknife=use_jackknife)
        print('\nTest Loss')
        print('SELD score (early stopping metric): {:0.4f} {}'.format(
            test_seld_scr[0] if use_jackknife else test_seld_scr,
            '[{:0.4f}, {:0.4f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
        print(
            'SED metrics: Error rate: {:0.4f} {}, F-score: {:0.3f} {}'.format(test_ER[0] if use_jackknife else test_ER,
                                                                              '[{:0.4f}, {:0.4f}]'.format(test_ER[1][0],
                                                                                                          test_ER[1][
                                                                                                              1]) if use_jackknife else '',
                                                                              100 * test_F[
                                                                                  0] if use_jackknife else 100 * test_F,
                                                                              '[{:0.4f}, {:0.4f}]'.format(
                                                                                  100 * test_F[1][0], 100 * test_F[1][
                                                                                      1]) if use_jackknife else ''))
        print('DOA metrics: Localization error: {:0.3f} {}, Localization Recall: {:0.3f} {}'.format(
            test_LE[0] if use_jackknife else test_LE,
            '[{:0.4f} , {:0.4f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else '',
            100 * test_LR[0] if use_jackknife else 100 * test_LR,
            '[{:0.4f}, {:0.4f}]'.format(100 * test_LR[1][0], 100 * test_LR[1][1]) if use_jackknife else ''))
        if params['average'] == 'macro':
            print('Classwise results on unseen test data')
            print('Class\tER\tF\tLE\tLR\tSELD_score')
            for cls_cnt in range(params['unique_classes']):
                print('{}\t{:0.5f} {}\t{:0.5f} {}\t{:0.5f} {}\t{:0.5f} {}\t{:0.5f} {}'.format(
                    cls_cnt,
                    classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt],
                    '[{:0.5f}, {:0.5f}]'.format(classwise_test_scr[1][0][cls_cnt][0],
                                                classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                    '[{:0.5f}, {:0.5f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                                classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                    '[{:0.5f}, {:0.5f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                                classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                    '[{:0.5f}, {:0.5f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                                classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                    '[{:0.5f}, {:0.5f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                                classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
