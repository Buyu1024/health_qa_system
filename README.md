# 🏥 医疗健康 RAG 智能问答系统

> 基于 RAG（检索增强生成）技术构建的医疗领域智能问答系统，支持多种疾病领域的权威指南查询和医疗咨询服务。

## ✨ 项目特色

- 🎯 **混合检索**：BM25 + BGE-M3 稠密向量 + Cross-Encoder 重排序
-  **意图识别**：BERT 查询分类器精准区分通用知识与医疗咨询
- ✅ **自验证机制**：Self-Verification Agent 降低幻觉问题
- 📊 **完整评估**：基于 Ragas 的专业评估体系
- 🚀 **生产级架构**：FastAPI + WebSocket 流式响应 + Redis 缓存

## 🛠️ 技术栈

| 类别 | 技术选型 |
|------|---------|
| **Web 框架** | FastAPI + Uvicorn + WebSocket |
| **向量数据库** | Milvus (health_rag collection) |
| **嵌入模型** | BGE-M3 (1024维) |
| **重排序模型** | BGE-Reranker-Large |
| **意图分类** | BERT-Base-Chinese (微调) |
| **大语言模型** | Qwen-Plus (DashScope API) |
| **缓存系统** | Redis |
| **数据存储** | MySQL |

## 📦 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/health_qa_system.git
cd health_qa_system
```

### 2. 配置敏感信息

**⚠️ 重要：保护您的 API 密钥和数据库密码**

本项目已采用安全的配置管理方式，敏感信息不会上传到 GitHub。您需要自行创建配置文件：

```bash
# 复制配置模板
cp config.example.ini config.ini

# 编辑配置文件，填入您的真实信息
# Windows 使用:
copy config.example.ini config.ini
```

然后在 `config.ini` 中修改以下敏感信息：

```ini
[mysql]
password = 您的MySQL密码

[redis]
password = 您的Redis密码

[llm]
dashscope_api_key = 您的DashScope API密钥
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载模型文件

项目中的大模型文件体积较大，未包含在 Git 仓库中。您需要手动下载或使用 ModelScope/HuggingFace 自动下载：

```python
# 项目会自动从配置路径加载模型
# 如需下载，请参考：
from modelscope import snapshot_download
snapshot_download('damo/nlp_bert_document-segmentation_chinese-base', 
                  local_dir='./rag_qa/models/nlp_bert_document-segmentation_chinese-base')
```

### 5. 启动服务

```bash
# 启动 FastAPI 服务
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 或直接运行主程序
python main.py
```

### 6. 访问系统

- API 文档：http://localhost:8000/docs
- WebSocket 测试：http://localhost:8000/static/index.html

## 📁 项目结构

```
health_qa_system/
├── rag_qa/                  # RAG 系统核心
│   ├── core/               # 核心逻辑
│   │   ├── rag_system.py          # RAG 主系统
│   │   ├── query_classifier.py    # BERT 查询分类器
│   │   ├── self_verification_agent.py  # 自验证 Agent
│   │   ├── vector_store.py        # 向量存储
│   │   └── document_processor.py  # 文档处理
│   ├── models/             # 模型文件
│   ├── data/               # 医疗文档数据
│   └── document_loaders/   # 文档加载器
├── mysql_qa/               # MySQL 和 BM25 检索
├── rag_assesment/          # RAG 评估系统
├── static/                 # 前端静态文件
├── api.py                  # FastAPI 接口
├── main.py                 # 主程序入口
├── config.example.ini      # 配置模板
└── .gitignore             # Git 忽略配置
```

## 🔒 安全说明

本项目严格遵循安全最佳实践：

1. **敏感信息保护**
   - `config.ini` 已加入 `.gitignore`，不会被上传到 GitHub
   - 提供了 `config.example.ini` 作为配置模板
   - 所有 API 密钥、密码等敏感信息需本地配置

2. **模型文件管理**
   - 大型模型文件已排除在 Git 仓库外
   - 建议使用 ModelScope 或 HuggingFace 自动下载

3. **数据隐私**
   - 数据库文件未包含在仓库中
   - 日志文件自动忽略

## 📊 项目成果

- ✅ 支持 **8 大疾病领域**权威指南查询
- ✅ 实现 **混合检索 + 重排序** 提升检索质量
- ✅ BERT 分类器准确率达 **99.9%**
- ✅ 基于 **Ragas** 的完整评估体系
- ✅ 生产级 **WebSocket 流式响应**

##  许可证

Apache License 2.0

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请提交 Issue 或发送邮件至 your-email@example.com

---

**⚠️ 注意事项**

1. 请勿将 `config.ini` 文件上传到公共仓库
2. 定期更新依赖包版本以修复安全漏洞
3. 生产环境建议使用环境变量管理敏感信息
4. 模型文件较大，建议使用 Git LFS 或单独下载
