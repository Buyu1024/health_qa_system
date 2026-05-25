from langchain.prompts import PromptTemplate

class VerificationPrompts:
    """自验证相关的 Prompt 模板"""
    
    @staticmethod
    def answer_verification_prompt():
        """答案验证 Prompt - 检查答案是否忠实于上下文"""
        return PromptTemplate(
            template="""
            你是一个严格的答案验证专家。你的任务是检查给定的答案是否完全基于提供的上下文，没有添加任何上下文中不存在的信息。
            
            **验证规则**：
            1. 答案中的每个事实陈述都必须能在上下文中找到明确支持
            2. 不能添加上下文中未提及的推断、解释或额外信息
            3. 如果答案包含上下文中没有的信息，必须标记为不忠实
            4. 数字、日期、专有名词等必须与上下文完全一致
            
            **上下文**：
            {context}
            
            **问题**：
            {question}
            
            **待验证的答案**：
            {answer}
            
            请按照以下格式输出验证结果：
            
            **分析过程**：
            [逐句分析答案中的每个陈述，说明它是否在上下文中得到支持]
            
            **忠实度评分**：
            [0-1之间的分数，1表示完全忠实，0表示完全不忠实]
            
            **验证结果**：
            [PASS/FAIL]
            
            **改进建议**：
            [如果验证失败，提供如何修改答案使其更忠实的建议；如果通过，写"无需修改"]
            """,
            input_variables=["context", "question", "answer"],
        )
    
    @staticmethod
    def answer_refinement_prompt():
        """答案优化 Prompt - 基于验证反馈改进答案"""
        return PromptTemplate(
            template="""
            你是一个专业的答案优化助手。根据验证反馈，重新生成一个更忠实于上下文的答案。
            
            **原始问题**：
            {question}
            
            **上下文**：
            {context}
            
            **原始答案**：
            {original_answer}
            
            **验证反馈**：
            {verification_feedback}
            
            **改进建议**：
            {improvement_suggestions}
            
            请生成一个新的答案，要求：
            1. 严格基于上下文，不添加任何额外信息
            2. 保持答案的准确性和完整性
            3. 使用清晰、简洁的语言
            4. 如果上下文不足以回答问题，明确说明
            
            **优化后的答案**：
            """,
            input_variables=["question", "context", "original_answer", 
                           "verification_feedback", "improvement_suggestions"],
        )
    
    @staticmethod
    def claim_extraction_prompt():
        """声明提取 Prompt - 从答案中提取可验证的声明"""
        return PromptTemplate(
            template="""
            从以下答案中提取所有独立的事实声明（claims），每行一个声明。
            
            **答案**：
            {answer}
            
            **提取的声明**：
            [每个声明应该是独立的、可验证的事实陈述]
            """,
            input_variables=["answer"],
        )
    
    @staticmethod
    def claim_verification_prompt():
        """声明验证 Prompt - 逐个验证声明是否得到上下文支持"""
        return PromptTemplate(
            template="""
            验证以下声明是否能从给定的上下文中得到支持。
            
            **上下文**：
            {context}
            
            **声明**：
            {claim}
            
            请回答：
            1. **是否支持**：[SUPPORTED/NOT_SUPPORTED/PARTIALLY_SUPPORTED]
            2. **理由**：[详细说明为什么支持或不支持，引用上下文中的相关部分]
            """,
            input_variables=["context", "claim"],
        )
