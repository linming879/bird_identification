def generate_bird_name_mapping_with_tit_group(
    bird_names: list[str],
    output_path: str = "bird_name_mapping.py",
    merge_tit_to_one: bool = True,
    tit_class_id: int = None  # 可自定义编号
):
    """
    生成 bird_name_mapping.py 文件，支持将所有包含 'tit' 的类别映射为同一类编号（可选）

    参数：
    - bird_names: 鸟类名称列表（List[str]）
    - output_path: 输出文件路径（.py）
    - merge_tit_to_one: 是否将所有 tit 鸟归为一类（True/False）
    - tit_class_id: 指定 tit 类别编号（默认自动选择最小可用值）
    """

    tit_names = [name for name in bird_names if "tit" in name.lower()]
    non_tit_names = sorted(set(name for name in bird_names if name not in tit_names))

    bird_name_mapping = {}

    if merge_tit_to_one:
        # 自动选择 tit 编号（最小未占用）
        if tit_class_id is None:
            tit_class_id = 0

        # 所有 tit 映射为 tit_class_id
        for name in sorted(set(tit_names)):
            bird_name_mapping[name] = tit_class_id

        # 其余类别从 tit_class_id + 1 起分配
        current_id = tit_class_id + 1
        for name in non_tit_names:
            while current_id == tit_class_id:
                current_id += 1
            bird_name_mapping[name] = current_id
            current_id += 1
    else:
        # 所有类别正常分配编号（不归类 tit）
        all_names = sorted(set(bird_names))
        bird_name_mapping = {name: idx for idx, name in enumerate(all_names)}

    # 保存为 .py 文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated bird name mapping\n")
        f.write("bird_name_mapping = {\n")
        for name, idx in bird_name_mapping.items():
            f.write(f'    "{name}": {idx},\n')
        f.write("}\n")

    print(f"✅ bird_name_mapping 已保存至: {output_path}")
    if merge_tit_to_one:
        print(f"📌 Tit 类别数: {len(tit_names)}，统一编号为: {tit_class_id}")
    else:
        print(f"📌 所有类别独立编码，共计: {len(bird_name_mapping)} 类别")

    return bird_name_mapping
