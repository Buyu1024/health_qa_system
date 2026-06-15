"""
清理微调模型中的训练中间文件

删除不必要的训练checkpoint和中间文件，只保留推理所需的模型文件。
这样可以大幅减小模型体积，且不影响推理性能。
"""

import os
import shutil


def clean_model_directory(model_path):
    """
    清理模型目录中的训练中间文件
    
    Args:
        model_path: 模型目录路径
    """
    print(f"🔍 开始清理模型目录: {model_path}")
    
    # 需要删除的文件和目录
    items_to_remove = [
        "bert_results",           # 整个训练结果目录（包含所有checkpoints）
        "optimizer.pt",           # 优化器状态
        "scheduler.pt",           # 学习率调度器
        "rng_state.pth",          # 随机数状态
        "trainer_state.json",     # 训练状态
        "training_args.bin",      # 训练参数
    ]
    
    removed_files = []
    saved_space = 0
    
    for item in items_to_remove:
        item_path = os.path.join(model_path, item)
        if os.path.exists(item_path):
            try:
                if os.path.isdir(item_path):
                    # 计算目录大小
                    dir_size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(item_path)
                        for filename in filenames
                    )
                    shutil.rmtree(item_path)
                    saved_space += dir_size
                    removed_files.append(f"{item} (目录, {dir_size/1024/1024:.2f} MB)")
                    print(f"✅ 已删除目录: {item} ({dir_size/1024/1024:.2f} MB)")
                else:
                    file_size = os.path.getsize(item_path)
                    os.remove(item_path)
                    saved_space += file_size
                    removed_files.append(f"{item} ({file_size/1024/1024:.2f} MB)")
                    print(f"✅ 已删除文件: {item} ({file_size/1024/1024:.2f} MB)")
            except Exception as e:
                print(f"❌ 删除失败 {item}: {e}")
        else:
            print(f"⚠️  不存在: {item}")
    
    # 统计结果
    print("\n" + "="*60)
    print("📊 清理完成统计")
    print("="*60)
    print(f"删除的项目数: {len(removed_files)}")
    print(f"节省空间: {saved_space/1024/1024:.2f} MB ({saved_space/1024/1024/1024:.2f} GB)")
    print("\n删除的项目:")
    for item in removed_files:
        print(f"  - {item}")
    
    # 显示剩余文件
    print("\n📁 保留的文件（推理必需）:")
    remaining_files = []
    for root, dirs, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, model_path)
            file_size = os.path.getsize(file_path)
            remaining_files.append((rel_path, file_size))
    
    total_remaining = sum(size for _, size in remaining_files)
    for rel_path, size in sorted(remaining_files):
        print(f"  - {rel_path}: {size/1024/1024:.2f} MB")
    
    print(f"\n总大小: {total_remaining/1024/1024:.2f} MB")
    print("="*60)


if __name__ == "__main__":
    model_path = r"D:\PythonProject\health_qa_system\rag_qa\models\bert_query_classifier"
    clean_model_directory(model_path)
