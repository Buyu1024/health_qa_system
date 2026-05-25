import os
import shutil
from pathlib import Path


def extract_documents(source_dir, target_dir):
    """
    从源目录及其子目录中提取所有文档到目标目录
    
    Args:
        source_dir: 源目录路径
        target_dir: 目标目录路径
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 支持的文档类型
    supported_extensions = {'.pdf', '.doc', '.docx', '.txt', '.md'}
    
    # 统计信息
    copied_count = 0
    skipped_count = 0
    
    # 遍历源目录及其子目录
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            file_path = Path(root) / filename
            
            # 检查文件扩展名是否支持
            if file_path.suffix.lower() in supported_extensions:
                target_path = Path(target_dir) / filename
                
                # 如果目标文件已存在，添加序号避免覆盖
                if target_path.exists():
                    counter = 1
                    stem = file_path.stem
                    suffix = file_path.suffix
                    while target_path.exists():
                        new_filename = f"{stem}_{counter}{suffix}"
                        target_path = Path(target_dir) / new_filename
                        counter += 1
                
                try:
                    shutil.copy2(file_path, target_path)
                    print(f"已复制: {filename}")
                    copied_count += 1
                except Exception as e:
                    print(f"复制失败 {filename}: {str(e)}")
                    skipped_count += 1
            else:
                skipped_count += 1
    
    print("\n" + "="*50)
    print(f"提取完成！")
    print(f"成功复制: {copied_count} 个文件")
    print(f"跳过: {skipped_count} 个文件")
    print(f"目标目录: {target_dir}")
    print("="*50)


if __name__ == '__main__':
    # 获取当前脚本所在目录的父目录
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # 源目录和目标目录
    source_directory = os.path.join(parent_dir, "卫健委发布的营养指南")
    target_directory = os.path.join(parent_dir, "data")
    
    print(f"源目录: {source_directory}")
    print(f"目标目录: {target_directory}")
    print("="*50)
    
    # 检查源目录是否存在
    if not os.path.exists(source_directory):
        print(f"错误: 源目录不存在 - {source_directory}")
    else:
        extract_documents(source_directory, target_directory)
