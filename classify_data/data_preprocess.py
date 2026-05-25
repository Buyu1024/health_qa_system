import json
import os
import random
from collections import Counter


def deduplicate_and_shuffle(input_file, output_file=None):
    """
    对JSON文件中的数据进行去重（按照query字段）并乱序
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出文件路径，默认为在原文件名后添加_processed后缀
    
    Returns:
        tuple: (去重前数量, 去重后数量, 输出文件路径)
    """
    # 获取输入文件的目录和文件名
    file_dir = os.path.dirname(input_file)
    file_name = os.path.basename(input_file)
    name_without_ext = os.path.splitext(file_name)[0]
    ext = os.path.splitext(file_name)[1]
    
    # 设置输出文件路径
    if output_file is None:
        output_file = os.path.join(file_dir, f"{name_without_ext}_processed{ext}")
    
    print(f"正在读取文件: {input_file}")
    
    # 读取所有数据
    data_list = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"警告: 跳过无效行 - {e}")
    
    original_count = len(data_list)
    print(f"原始数据量: {original_count} 条")
    
    # 去重：按照query字段去重，保留第一条出现的记录
    seen_queries = set()
    unique_data = []
    duplicate_count = 0
    
    for item in data_list:
        query = item.get('query', '')
        if query not in seen_queries:
            seen_queries.add(query)
            unique_data.append(item)
        else:
            duplicate_count += 1
    
    after_dedup_count = len(unique_data)
    print(f"去重后数据量: {after_dedup_count} 条")
    print(f"去除重复数据: {duplicate_count} 条")
    
    # 统计label类型的数量
    label_counter = Counter(item.get('label', '未知') for item in unique_data)
    print(f"\nLabel类型统计（共 {len(label_counter)} 种类型）:")
    print("-" * 50)
    for label, count in sorted(label_counter.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / after_dedup_count) * 100
        print(f"  {label}: {count} 条 ({percentage:.2f}%)")
    print("-" * 50)
    
    # 乱序：随机打乱数据顺序
    random.shuffle(unique_data)
    print("\n数据已随机打乱")
    
    # 写入输出文件
    print(f"正在写入文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in unique_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"处理完成！")
    print(f"输出文件: {output_file}")
    
    return original_count, after_dedup_count, output_file


if __name__ == '__main__':
    # 输入文件路径
    input_file = os.path.join(os.path.dirname(__file__), 'medical_qa.json')
    
    # 执行去重和乱序
    try:
        original_count, final_count, output_file = deduplicate_and_shuffle(input_file)
        print("\n✓ 处理成功！")
        print(f"  原始数据: {original_count} 条")
        print(f"  处理后数据: {final_count} 条")
        print(f"  输出文件: {output_file}")
    except Exception as e:
        print(f"\n✗ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
