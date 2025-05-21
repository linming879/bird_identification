"""
工具函数：用于根据 bird_name_mapping 构建 index_to_bird 和 class_labels 映射。
适用于普通实验和合并类（如 tit 合并）实验。
"""

from collections import defaultdict

def build_index_to_label(bird_name_mapping: dict, merge_names: bool = True) -> dict:
    """
    构建 index -> bird label 的映射字典。

    参数:
        bird_name_mapping (dict): 鸟名到索引的映射
        merge_names (bool): 是否合并多个 bird name 到一个 label（用于合并类别场景）

    返回:
        dict: index -> bird label（字符串）
    """
    reverse_map = defaultdict(list)
    for name, idx in bird_name_mapping.items():
        reverse_map[idx].append(name)

    if merge_names:
        # 将多个名字拼接（如合并了多个 tit 类）
        return {idx: " / ".join(sorted(names)) for idx, names in reverse_map.items()}
    else:
        # 只取第一个名字（用于非合并场景）
        return {idx: names[0] for idx, names in reverse_map.items()}


def get_class_labels(bird_name_mapping: dict, merge_names: bool = True) -> list:
    """
    返回与类别数量一致的 class_labels 列表，用于混淆矩阵等可视化。

    参数:
        bird_name_mapping (dict): 鸟名到索引的映射
        merge_names (bool): 是否合并多个名字

    返回:
        list: index 顺序排列的标签名
    """
    index_to_bird = build_index_to_label(bird_name_mapping, merge_names)
    num_classes = len(set(bird_name_mapping.values()))
    return [index_to_bird[i] for i in range(num_classes)]
