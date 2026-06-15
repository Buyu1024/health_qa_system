import re
from base.logger import logger
from verification_prompts import VerificationPrompts

class SelfVerificationPipeline:
    """
    自验证流水线 - 通过多轮 Prompt Engineering 提高 RAG 系统答案的忠实度

    工作流程：
    1. 生成初始答案
    2. 提取答案中的声明
    3. 验证每个声明是否得到上下文支持
    4. 如果验证失败，优化答案
    5. 返回最终答案和验证报告
    """

    def __init__(self, llm, max_refinement_iterations=2, verification_threshold=0.9):
        """
        初始化自验证流水线

        Args:
            llm: 语言模型实例
            max_refinement_iterations: 最大优化迭代次数
            verification_threshold: 验证通过阈值（0-1）
        """
        self.llm = llm
        self.max_refinement_iterations = max_refinement_iterations
        self.verification_threshold = verification_threshold
        self.verification_prompt = VerificationPrompts.answer_verification_prompt()
        self.refinement_prompt = VerificationPrompts.answer_refinement_prompt()
        self.claim_extraction_prompt = VerificationPrompts.claim_extraction_prompt()
        self.claim_verification_prompt = VerificationPrompts.claim_verification_prompt()

    def _collect_llm_output(self, prompt_input):
        """收集 LLM 的完整输出"""
        try:
            output_parts = []
            for token in self.llm(prompt_input):
                output_parts.append(token)
            return ''.join(output_parts).strip()
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return ""

    def extract_claims(self, answer):
        """
        从答案中提取独立的声明

        Args:
            answer: 待分析的答案

        Returns:
            list: 声明列表
        """
        logger.info("正在从答案中提取声明...")
        prompt_input = self.claim_extraction_prompt.format(answer=answer)
        claims_text = self._collect_llm_output(prompt_input)

        # 解析声明（每行一个）
        claims = [claim.strip() for claim in claims_text.split('\n')
                 if claim.strip() and not claim.startswith('**')]

        logger.info(f"提取到 {len(claims)} 个声明")
        return claims

    def verify_claims(self, context, claims):
        """
        逐个验证声明是否得到上下文支持

        Args:
            context: 参考上下文
            claims: 待验证的声明列表

        Returns:
            dict: 验证结果，包含每个声明的支持状态
        """
        logger.info(f"正在验证 {len(claims)} 个声明...")
        verification_results = []

        for i, claim in enumerate(claims):
            logger.info(f"验证声明 {i+1}/{len(claims)}: {claim[:50]}...")
            prompt_input = self.claim_verification_prompt.format(
                context=context,
                claim=claim
            )
            result_text = self._collect_llm_output(prompt_input)

            # 解析验证结果
            is_supported = "SUPPORTED" in result_text.upper()
            is_partially = "PARTIALLY_SUPPORTED" in result_text.upper()

            verification_results.append({
                'claim': claim,
                'supported': is_supported,
                'partially_supported': is_partially,
                'reason': result_text
            })

        return verification_results

    def verify_answer(self, context, question, answer):
        """
        验证答案的忠实度

        Args:
            context: 参考上下文
            question: 原始问题
            answer: 待验证的答案

        Returns:
            dict: 验证结果，包含评分、是否通过、分析过程等
        """
        logger.info("开始验证答案忠实度...")
        prompt_input = self.verification_prompt.format(
            context=context,
            question=question,
            answer=answer
        )

        verification_text = self._collect_llm_output(prompt_input)

        # 解析验证结果
        faithfulness_score = self._extract_faithfulness_score(verification_text)
        is_pass = "PASS" in verification_text.upper()

        # 提取改进建议
        improvement_suggestions = self._extract_improvement_suggestions(verification_text)

        result = {
            'score': faithfulness_score,
            'pass': is_pass,
            'analysis': verification_text,
            'improvement_suggestions': improvement_suggestions
        }

        logger.info(f"验证完成 - 评分: {faithfulness_score:.2f}, 结果: {'PASS' if is_pass else 'FAIL'}")
        return result

    def _extract_faithfulness_score(self, verification_text):
        """从验证文本中提取忠实度评分"""
        # 尝试匹配各种格式的评分
        patterns = [
            r'忠实度评分[：:]\s*([0-9.]+)',
            r'faithfulness\s*score[：:]\s*([0-9.]+)',
            r'评分[：:]\s*([0-9.]+)',
            r'score[：:]\s*([0-9.]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, verification_text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return min(max(score, 0.0), 1.0)  # 限制在 0-1 范围
                except ValueError:
                    continue

        # 如果无法提取，根据 PASS/FAIL 推断
        if "PASS" in verification_text.upper():
            return 0.9
        else:
            return 0.3

    def _extract_improvement_suggestions(self, verification_text):
        """从验证文本中提取改进建议"""
        patterns = [
            r'改进建议[：:]([\s\S]*?)(?:\n\n|\Z)',
            r'improvement\s*suggestions[：:]([\s\S]*?)(?:\n\n|\Z)',
            r'建议[：:]([\s\S]*?)(?:\n\n|\Z)',
        ]

        for pattern in patterns:
            match = re.search(pattern, verification_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return "无需修改" if "PASS" in verification_text.upper() else "需要重新生成更忠实的答案"

    def refine_answer(self, question, context, original_answer, verification_result):
        """
        基于验证反馈优化答案

        Args:
            question: 原始问题
            context: 参考上下文
            original_answer: 原始答案
            verification_result: 验证结果

        Returns:
            str: 优化后的答案
        """
        logger.info("开始优化答案...")
        prompt_input = self.refinement_prompt.format(
            question=question,
            context=context,
            original_answer=original_answer,
            verification_feedback=verification_result['analysis'],
            improvement_suggestions=verification_result['improvement_suggestions']
        )

        refined_answer = self._collect_llm_output(prompt_input)
        logger.info(f"答案优化完成，新答案长度: {len(refined_answer)} 字符")
        return refined_answer

    def generate_verified_answer(self, question, context, initial_answer=None, generate_initial=True):
        """
        生成经过验证的答案（主方法）

        Args:
            question: 用户问题
            context: 检索到的上下文
            initial_answer: 初始答案（如果为 None 且 generate_initial=True，则先生成）
            generate_initial: 是否需要先生成初始答案

        Returns:
            dict: 包含最终答案、验证结果和优化历史
        """
        result_history = []

        # Step 1: 生成初始答案（如果需要）
        if initial_answer is None and generate_initial:
            logger.info("生成初始答案...")
            from prompts import RAGPrompts
            rag_prompt = RAGPrompts.rag_prompt()
            prompt_input = rag_prompt.format(
                context=context,
                question=question,
                history="",
                phone=""
            )
            initial_answer = self._collect_llm_output(prompt_input)

        current_answer = initial_answer
        iteration = 0

        # Step 2-4: 验证和优化循环
        while iteration < self.max_refinement_iterations:
            iteration += 1
            logger.info(f"=== 验证迭代 {iteration} ===")

            # 验证当前答案
            verification_result = self.verify_answer(context, question, current_answer)

            result_history.append({
                'iteration': iteration,
                'answer': current_answer,
                'verification': verification_result
            })

            # 检查是否通过验证
            if (verification_result['score'] >= self.verification_threshold and
                verification_result['pass']):
                logger.info(f"✓ 答案通过验证（评分: {verification_result['score']:.2f}）")
                break

            # 如果未通过且还有迭代次数，进行优化
            if iteration < self.max_refinement_iterations:
                logger.info(f"✗ 答案未通过验证，进行优化...")
                current_answer = self.refine_answer(
                    question, context, current_answer, verification_result
                )
            else:
                logger.warning(f"达到最大迭代次数，使用最后一次生成的答案")

        # 返回最佳结果
        best_result = result_history[-1]

        return {
            'final_answer': best_result['answer'],
            'verification_result': best_result['verification'],
            'iterations': iteration,
            'history': result_history,
            'faithfulness_score': best_result['verification']['score']
        }
