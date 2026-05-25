"""
RAG 系统批量测试工具 - 生成 RAGAS 兼容格式
直接调用 RAG 系统进行批量问答测试，生成可用于 RAGAS 评估的数据文件
输出格式与 rag_evaluate_data.json 完全一致
"""
import json
import time
from datetime import datetime
from pathlib import Path
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from main import IntergrateQASystem


class RAGBatchTesterForRagas:
    """RAG 系统批量测试器 - RAGAS 兼容格式"""
    
    def __init__(self):
        """初始化测试器"""
        print("🚀 正在初始化 RAG 系统...")
        self.qa_system = IntergrateQASystem()
        print("✅ RAG 系统初始化完成\n")
        self.results = []
        
    def load_test_questions(self, data_path=None):
        """加载测试问题（包含 ground_truth）
        
        Args:
            data_path: 测试数据文件路径，默认为 rag_assesment/rag_evaluate_data.json
            
        Returns:
            list: 测试数据列表，每个元素包含 question 和 ground_truth
        """
        if data_path is None:
            data_path = os.path.join(project_root, "rag_assesment", "rag_evaluate_data.json")
        
        print(f"📂 正在加载测试数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取问题和标准答案
        test_items = []
        for item in data:
            test_items.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"]
            })
        
        print(f"✅ 成功加载 {len(test_items)} 个测试样本\n")
        return test_items
    
    def get_retrieved_context(self, question):
        """获取 RAG 系统检索到的上下文
        
        Args:
            question: 问题文本
            
        Returns:
            list: 检索到的文档片段列表
        """
        try:
            # 直接调用 RAG 系统的检索功能
            # 这里需要访问 rag_system 的内部方法
            rag_system = self.qa_system.rag_system
            
            # 使用 _retrieve_and_merge 方法获取上下文文档
            context_docs = rag_system._retrieve_and_merge(question, source_filter=None)
            
            # 提取文档内容
            contexts = [doc.page_content for doc in context_docs]
            
            return contexts
        except Exception as e:
            print(f"⚠️  检索上下文失败: {e}")
            return []
    
    def generate_answer(self, question, session_id=None):
        """生成答案
        
        Args:
            question: 问题文本
            session_id: 会话 ID
            
        Returns:
            str: 生成的答案
        """
        answer = ""
        try:
            # 调用 RAG 系统获取答案（流式）
            for token, is_complete in self.qa_system.query(
                query=question,
                source_filter=None,
                session_id=session_id
            ):
                if token:
                    answer += token
                    
            return answer
        except Exception as e:
            print(f"❌ 生成答案失败: {e}")
            return ""
    
    def test_single_question(self, item, index):
        """测试单个问题并生成 RAGAS 格式结果
        
        Args:
            item: 测试项，包含 question 和 ground_truth
            index: 问题索引
            
        Returns:
            dict: RAGAS 格式的测试结果
        """
        question = item["question"]
        ground_truth = item["ground_truth"]
        session_id = f"ragas-test-{index}"
        
        print(f"[{index}] 问题: {question[:60]}...")
        
        start_time = time.time()
        
        # 步骤 1: 获取检索到的上下文
        print(f"   🔍 正在检索上下文...")
        contexts = self.get_retrieved_context(question)
        print(f"   ✅ 检索到 {len(contexts)} 个文档片段")
        
        # 步骤 2: 生成答案
        print(f"   💬 正在生成答案...")
        answer = self.generate_answer(question, session_id)
        
        processing_time = time.time() - start_time
        
        # 构建 RAGAS 格式的结果
        result = {
            "question": question,
            "context": contexts,  # 必须是数组格式
            "answer": answer,
            "ground_truth": ground_truth
        }
        
        print(f"   ⏱️  处理时间: {processing_time:.2f} 秒")
        print(f"   📝 答案长度: {len(answer)} 字符")
        print(f"   ✓ 完成\n")
        
        return result
    
    def batch_test(self, test_items, delay=1.0):
        """批量测试所有问题
        
        Args:
            test_items: 测试数据列表
            delay: 每个问题之间的延迟时间（秒）
            
        Returns:
            list: RAGAS 格式的测试结果
        """
        total = len(test_items)
        print("=" * 80)
        print(f"📊 开始批量测试 - 共 {total} 个问题")
        print(f"📋 输出格式: RAGAS 兼容格式 (question, context, answer, ground_truth)")
        print("=" * 80 + "\n")
        
        for i, item in enumerate(test_items, 1):
            try:
                result = self.test_single_question(item, i)
                self.results.append(result)
                
                # 延迟，避免请求过快
                if i < total and delay > 0:
                    print(f"   ⏳ 等待 {delay} 秒...\n")
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"   ❌ 测试失败: {e}\n")
                # 添加失败记录
                self.results.append({
                    "question": item["question"],
                    "context": [],
                    "answer": "",
                    "ground_truth": item["ground_truth"],
                    "error": str(e)
                })
        
        return self.results
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 80)
        print("📈 测试摘要")
        print("=" * 80)
        
        total = len(self.results)
        success_count = sum(1 for r in self.results if r.get("answer", ""))
        failed_count = total - success_count
        
        print(f"✅ 总测试数: {total}")
        print(f"✅ 成功数: {success_count}")
        print(f"❌ 失败数: {failed_count}")
        
        if success_count > 0:
            avg_context_count = sum(len(r["context"]) for r in self.results if r.get("answer")) / success_count
            avg_answer_length = sum(len(r["answer"]) for r in self.results if r.get("answer")) / success_count
            
            print(f"📚 平均检索文档数: {avg_context_count:.1f} 个/问题")
            print(f"📝 平均答案长度: {avg_answer_length:.0f} 字符/问题")
        
        print(f"\n💡 提示: 结果已保存为 RAGAS 兼容格式，可直接用于 RAGAS 评估")
    
    def save_ragas_format(self, filename="rag_evaluate_results.json"):
        """保存结果为 RAGAS 兼容格式
        
        Args:
            filename: 输出文件名
        """
        output_path = os.path.join(project_root, "rag_assesment", filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 过滤掉失败的测试（可选）
        valid_results = [r for r in self.results if r.get("answer", "")]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(valid_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 RAGAS 格式结果已保存到: {output_path}")
        print(f"📊 有效测试结果: {len(valid_results)} 个")
        
        return output_path
    
    def close(self):
        """关闭系统连接"""
        if hasattr(self.qa_system, 'mysql_client'):
            self.qa_system.mysql_client.close()
            print("\n🔌 数据库连接已关闭")


def main():
    """主函数"""
    tester = RAGBatchTesterForRagas()
    
    try:
        # 加载测试问题
        test_items = tester.load_test_questions()
        
        # 执行批量测试（延迟 1 秒）
        tester.batch_test(test_items, delay=1.0)
        
        # 打印摘要
        tester.print_summary()
        
        # 保存为 RAGAS 格式
        output_file = tester.save_ragas_format("rag_evaluate_results.json")
        
        print("\n" + "=" * 80)
        print("🎯 下一步操作")
        print("=" * 80)
        print(f"1. 检查生成的文件: {output_file}")
        print(f"2. 运行 RAGAS 评估:")
        print(f"   python rag_assesment/rag_as.py")
        print(f"3. 查看评估报告")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断测试")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.close()


if __name__ == "__main__":
    print("=" * 80)
    print("🧪 RAG 系统批量测试工具 - RAGAS 兼容格式")
    print("=" * 80)
    print("本工具将:")
    print("  1. 调用 RAG 系统回答测试问题")
    print("  2. 记录检索到的上下文文档")
    print("  3. 生成与 rag_evaluate_data.json 相同格式的文件")
    print("  4. 可直接用于 RAGAS 评估")
    print("=" * 80 + "\n")
    
    main()
