"""
批量处理PDF文档，提取文本内容
支持文字版PDF直接提取，扫描版PDF需要OCR支持
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.document_loaders import extract_text


def batch_process_pdfs(source_folder, output_folder):
    """
    批量处理PDF文件
    
    Args:
        source_folder: 源文件夹路径（包含子文件夹）
        output_folder: 输出文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 统计信息
    total_count = 0
    success_count = 0
    failed_count = 0
    
    print(f"开始处理文件夹: {source_folder}")
    print(f"输出目录: {output_folder}")
    print("="*60)
    
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                total_count += 1
                pdf_path = os.path.join(root, filename)
                
                # 创建输出文件名（保留子文件夹结构）
                relative_path = os.path.relpath(root, source_folder)
                if relative_path == '.':
                    output_subfolder = output_folder
                else:
                    output_subfolder = os.path.join(output_folder, relative_path)
                    os.makedirs(output_subfolder, exist_ok=True)
                
                # 生成txt文件名
                txt_filename = filename.replace('.pdf', '.txt')
                txt_path = os.path.join(output_subfolder, txt_filename)
                
                print(f"\n[{total_count}] 处理: {filename}")
                print(f"    来源: {os.path.relpath(pdf_path, parent_dir)}")
                
                try:
                    # 提取文本
                    content = extract_text(pdf_path)
                    
                    # 保存到txt文件
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(f"来源文件: {filename}\n")
                        f.write("="*60 + "\n\n")
                        f.write(content)
                    
                    # 显示统计信息
                    char_count = len(content)
                    print(f"    ✓ 成功提取 {char_count} 个字符")
                    print(f"    保存至: {os.path.relpath(txt_path, parent_dir)}")
                    success_count += 1
                    
                except Exception as e:
                    print(f"    ✗ 处理失败: {str(e)}")
                    failed_count += 1
    
    # 打印总结
    print("\n" + "="*60)
    print("处理完成！")
    print(f"总文件数: {total_count}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"输出目录: {output_folder}")
    print("="*60)


if __name__ == "__main__":
    # 源文件夹和目标文件夹
    source_folder = os.path.join(parent_dir, "卫健委发布的营养指南")
    output_folder = os.path.join(parent_dir, "data", "processed_texts")
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"错误: 源文件夹不存在 - {source_folder}")
        sys.exit(1)
    
    # 开始批量处理
    batch_process_pdfs(source_folder, output_folder)
