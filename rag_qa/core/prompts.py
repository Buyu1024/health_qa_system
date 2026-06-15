from langchain.prompts import PromptTemplate

class RAGPrompts:
    # @staticmethod
    # def rag_prompt():
    #     return PromptTemplate(
    #         template="""
    #         你是一个智能助手，帮助用户回答问题。
    #         如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
    #         如果答案来源于检索到的文档，请在回答中说明。
    #
    #         上下文: {context}
    #         问题: {question}
    #
    #         如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
    #         回答:
    #         """,
    #         input_variables = ["context","question","phone"],
    #     )
    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""
        你是一个智能助手，负责帮助用户回答问题。请按照以下步骤处理：

        1. **分析问题和上下文**：
           - 基于提供的上下文（如果有）和你的知识回答问题。
           - 如果答案来源于检索到的文档，请在回答中明确说明，例如：“根据提供的文档，……”。

        2. **评估对话历史**：
           - 检查对话历史是否与当前问题相关（例如，是否涉及相同的话题、实体或问题背景）。
           - 如果对话历史与问题相关，请结合历史信息生成更准确的回答。
           - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和问题回答。

        3. **生成回答**：
           - 提供清晰、准确的回答，避免无关信息。
           - 如果上下文和历史消息均不足以回答问题，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”

        **上下文**: {context}
        **对话历史**:
        {history}
        **问题**: {question}

        **回答**:
        """,
            input_variables=["context", "history", "question", "phone"],
        )
    @staticmethod
    def hyde_prompt():
        return PromptTemplate(
            template="""
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            问题: {query}  
            假设答案:
            """,
            input_variables=["query"],
        )
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询：  
            查询: {query}  
            子查询:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )
    @staticmethod
    def backtracking_prompt():
        """
        回溯问题 Prompt：将复杂或模糊的查询简化为核心问题
        
        例如：
        - 原始："我最近总是头晕，血压有点高，是不是得了高血压？"
        - 简化："高血压的主要症状有哪些？"
        """
        return PromptTemplate(
            template="""
            你是一个问题简化专家。请将用户的复杂或模糊查询简化为一个更基础、更易检索的核心问题。
            
            简化原则：
            1. 保留核心意图，去除冗余信息
            2. 转化为标准的问句形式
            3. 只返回简化后的问题，不要任何解释或额外内容
            
            原始查询: {query}
            
            简化后的问题:
            """,
            input_variables=["query"],
        )


