"""
RAGAS 评估结果数据处理脚本
从 CSV 文件中提取评估指标并打印到控制台
"""
import pandas as pd
import ast
import os

# ANSI 颜色代码
RED = '\033[91m'      # 红色 - 低于 0.9
GREEN = '\033[92m'    # 绿色 - 0.9 及以上
YELLOW = '\033[93m'   # 黄色 - 警告
RESET = '\033[0m'     # 重置颜色
BOLD = '\033[1m'      # 加粗

def colorize_value(value, threshold=0.9):
    """根据阈值给数值添加颜色"""
    if value < threshold:
        return f"{RED}{value:.4f}{RESET}"
    else:
        return f"{GREEN}{value:.4f}{RESET}"


def data_process(csv_file):
    try:
        # 读取 CSV 文件
        df = pd.read_csv(csv_file)

        print(f"✅ 成功读取 CSV 文件")
        print(f"📋 列名: {df.columns.tolist()}")
        print(f"📏 行数: {len(df)}\n")

        # 提取 scores 列的内容
        if 'scores' in df.columns:
            scores_str = df['scores'].iloc[0]  # 获取第一行的 scores 列

            print("=" * 80)
            print("📈 原始 scores 数据（字符串格式）:")
            print("=" * 80)
            print(f"{scores_str[:200]}...\n")  # 打印前200个字符

            # 将字符串转换为 Python 列表
            try:
                scores_list = ast.literal_eval(scores_str)

                print("=" * 80)
                print("✅ 成功解析为列表")
                print(f"📊 共 {len(scores_list)} 个样本的评估结果\n")

                # 打印表头
                print("=" * 120)
                print(
                    f"{'样本':^6} | {'Faithfulness':^16} | {'Answer Relevancy':^18} | {'Context Precision':^18} | {'Context Recall':^16}")
                print("=" * 120)

                # 每行输出一个样本的四个指标
                for i, score in enumerate(scores_list, 1):
                    faithfulness = score['faithfulness']
                    answer_relevancy = score['answer_relevancy']
                    context_precision = score['context_precision']
                    context_recall = score['context_recall']

                    # 为低于 0.9 的值添加红色标记
                    f_str = colorize_value(faithfulness)
                    ar_str = colorize_value(answer_relevancy)
                    cp_str = colorize_value(context_precision)
                    cr_str = colorize_value(context_recall)

                    print(f"{i:^6} | {f_str:^16} | {ar_str:^18} | {cp_str:^18} | {cr_str:^16}")

                print("=" * 120)

                print("\n" + "=" * 120)
                print("📊 统计摘要（平均值）")
                print("=" * 120)

                avg_faithfulness = sum(s['faithfulness'] for s in scores_list) / len(scores_list)
                avg_answer_relevancy = sum(s['answer_relevancy'] for s in scores_list) / len(scores_list)
                avg_context_precision = sum(s['context_precision'] for s in scores_list) / len(scores_list)
                avg_context_recall = sum(s['context_recall'] for s in scores_list) / len(scores_list)

                # 为平均值添加颜色
                avg_f_str = colorize_value(avg_faithfulness)
                avg_ar_str = colorize_value(avg_answer_relevancy)
                avg_cp_str = colorize_value(avg_context_precision)
                avg_cr_str = colorize_value(avg_context_recall)

                print(
                    f"{BOLD}{'平均':^6}{RESET} | {avg_f_str:^16} | {avg_ar_str:^18} | {avg_cp_str:^18} | {avg_cr_str:^16}")
                print("=" * 120)

                # 统计低分样本
                low_faithfulness = [i + 1 for i, s in enumerate(scores_list) if s['faithfulness'] < 0.9]
                low_answer_relevancy = [i + 1 for i, s in enumerate(scores_list) if s['answer_relevancy'] < 0.9]
                low_context_precision = [i + 1 for i, s in enumerate(scores_list) if s['context_precision'] < 0.9]
                low_context_recall = [i + 1 for i, s in enumerate(scores_list) if s['context_recall'] < 0.9]

                print("\n" + "=" * 120)
                print(f"{YELLOW}⚠️  需要关注的样本（指标 < 0.9）{RESET}")
                print("=" * 120)

                if low_faithfulness:
                    print(f"{RED}• Faithfulness < 0.9: 样本 {low_faithfulness}{RESET}")
                if low_answer_relevancy:
                    print(f"{RED}• Answer Relevancy < 0.9: 样本 {low_answer_relevancy}{RESET}")
                if low_context_precision:
                    print(f"{RED}• Context Precision < 0.9: 样本 {low_context_precision}{RESET}")
                if low_context_recall:
                    print(f"{RED}• Context Recall < 0.9: 样本 {low_context_recall}{RESET}")

                if not any([low_faithfulness, low_answer_relevancy, low_context_precision, low_context_recall]):
                    print(f"{GREEN}✅ 所有样本的所有指标均 >= 0.9，表现优秀！{RESET}")

                print("\n" + "=" * 120)
                print(f"{BOLD}💡 图例说明{RESET}")
                print("=" * 120)
                print(f"  {GREEN}■ 绿色{RESET}: 指标 >= 0.9（良好）")
                print(f"  {RED}■ 红色{RESET}: 指标 < 0.9（需要优化）")
                print("\n" + "=" * 120)
                print(f"{BOLD}📊 指标说明{RESET}")
                print("=" * 120)
                print("• Faithfulness (忠实度): 答案是否基于检索到的上下文（检测幻觉）")
                print("• Answer Relevancy (相关性): 答案与问题的相关程度")
                print("• Context Precision (精确度): 检索到的上下文中有多少是相关的")
                print("• Context Recall (召回率): 检索到的上下文是否包含了回答问题所需的信息")
                print("• 所有指标范围: 0-1，越高越好")
                print("• 建议阈值: >= 0.9 为优秀，0.7-0.9 为良好，< 0.7 需要优化")

            except Exception as e:
                print(f"❌ 解析 scores 数据失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ CSV 文件中未找到 'scores' 列")
            print(f"可用的列: {df.columns.tolist()}")

    except FileNotFoundError:
        print(f"❌ 文件不存在: {csv_file}")
        print(f"请检查文件路径是否正确")
    except Exception as e:
        print(f"❌ 处理文件时发生错误: {e}")
        import traceback
        traceback.print_exc()
    print("\n" + "=" * 80)

if __name__ == '__main__':
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # CSV 文件路径
    csv_file = os.path.join(current_dir, "自验证RAG", "rag_evaluate_result.csv")
    print("=" * 80)
    print("📊 RAGAS 评估结果数据处理")
    print("=" * 80)
    print(f"\n📂 正在读取文件: {csv_file}\n")
    data_process(csv_file)
