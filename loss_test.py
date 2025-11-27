def generate_new_labels(raw_data_list, max_num_events=5):  # 固定max_num_events=5
    """
    输入：raw_data_list → 原始数据（每帧事件列表）
    输出：new_labels → 适配5个声源的新标签
    """
    new_labels = []
    for raw_data in raw_data_list:
        T = raw_data["frames"]
        frame_events = raw_data["frame_events"]  # 每帧事件列表（长度≤5）

        # 初始化：[T,5,17]（5个声源位）、[T,5]（掩码）
        events = torch.zeros((T, max_num_events, 17))
        event_masks = torch.zeros((T, max_num_events))

        for t in range(T):
            events_t = frame_events[t]  # 第t帧真实声源（0~5个）
            e_num = len(events_t)
            if e_num == 0:
                continue

            # 填充真实声源信息（类别+DOA+有效性）
            for e in range(e_num):
                evt = events_t[e]
                events[t, e, 0:13] = torch.tensor(evt["classes"])  # 多标签类别
                events[t, e, 13:16] = torch.tensor(evt["doa"])  # DOA坐标
                events[t, e, 16] = 1.0  # 标记有效
                event_masks[t, e] = 1.0  # 掩码有效

        new_labels.append({
            "frames": T,
            "events": events,
            "event_masks": event_masks
        })
    return new_labels