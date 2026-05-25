"""
自验证代理使用示例
展示如何使用 SelfVerificationAgent 提高答案质量
"""
import os
import sys

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from langchain_community.llms import Tongyi
from base.config import Config
from core.self_verification_agent import SelfVerificationAgent

# 初始化配置和 LLM
conf = Config()
llm = Tongyi(
    model_name="qwen-plus",
    dashscope_api_key=conf.DASHSCOPE_API_KEY
)

print("=" * 80)
print("自验证代理使用示例")
print("=" * 80)

# 创建自验证代理
agent = SelfVerificationAgent(
    llm=llm,
    max_refinement_iterations=2,
    verification_threshold=0.8
)

# 示例 1: 完整的验证流程
print("\n【示例 1】完整的验证流程")
print("-" * 80)

question = "儿童青少年肥胖食养建议每天摄入多少种以上食物？"
context = "日常膳食做到食物多样，每天的食物应包括谷薯类、蔬菜水果、禽畜鱼蛋奶类和大豆坚果类；达到每天摄入 12 种以上食物，每周摄入 25 种以上食物。"

print(f"问题: {question}")
print(f"上下文: {context[:100]}...")
print("\n正在生成并验证答案...\n")

result = agent.generate_verified_answer(
    question=question,
    context=context,
    generate_initial=True
)

print(f"✓ 最终答案: {result['final_answer']}")
print(f"✓ 忠实度评分: {result['faithfulness_score']:.2f}")
print(f"✓ 迭代次数: {result['iterations']}")
print(f"✓ 验证结果: {'通过' if result['verification_result']['pass'] else '未通过'}")

# 示例 2: 仅验证已有答案
print("\n\n【示例 2】仅验证已有答案")
print("-" * 80)

existing_answer = "儿童青少年肥胖食养建议每天摄入 12 种以上食物。"
print(f"问题: {question}")
print(f"待验证答案: {existing_answer}")
print("\n正在验证...\n")

verification = agent.verify_answer(
    context=context,
    question=question,
    answer=existing_answer
)

print(f"验证评分: {verification['score']:.2f}")
print(f"验证结果: {'通过 ✓' if verification['pass'] else '未通过 ✗'}")
print(f"\n分析过程:\n{verification['analysis'][:500]}...")

# 示例 3: 提取和验证实例
print("\n\n【示例 3】提取和验证实例")
print("-" * 80)

answer = "学龄前儿童（2~5 岁）每天应摄入 350~500mL 或相当量的奶及奶制品。"
context2 = "学龄前儿童（2~5 岁）每天摄入 350~500mL 或相当量的奶及奶制品。学龄儿童（6~17 岁）每天摄入 300mL 以上或相当量的奶及奶制品。"

print(f"答案: {answer}")
print("\n步骤 1: 提取声明")
claims = agent.extract_claims(answer)
for i, claim in enumerate(claims, 1):
    print(f"  声明 {i}: {claim}")

print("\n步骤 2: 验证每个声明")
verifications = agent.verify_claims(context2, claims)
for i, v in enumerate(verifications, 1):
    status = "✓ 支持" if v['supported'] else ("△ 部分支持" if v['partially_supported'] else "✗ 不支持")
    print(f"\n  声明 {i}: {status}")
    print(f"  理由: {v['reason'][:200]}...")

print("\n" + "=" * 80)
print("示例完成！")
print("=" * 80)
