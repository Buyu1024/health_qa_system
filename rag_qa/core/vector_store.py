import os,sys

import torch
import numpy as np
# 导入 BGE-M3 嵌入函数，用于生成文档和查询的向量表示
from milvus_model.hybrid import BGEM3EmbeddingFunction
# 导入 Milvus 相关类，用于操作向量数据库
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker
# 导入 Document 类，用于创建文档对象
from langchain.docstore.document import Document
# 导入 CrossEncoder，用于重排序和 NLI 判断
from sentence_transformers import CrossEncoder
# 导入 hashlib 模块，用于生成唯一 ID 的哈希值
import hashlib

from base.logger import logger
from base.config import Config

conf = Config()

class VectorStore:
    def __init__(self):
        # 设置 Milvus 集合名称
        self.collection_name = conf.MILVUS_COLLECTION_NAME
        # 设置 Milvus 主机地址
        self.host = conf.MILVUS_HOST
        # 设置 Milvus 端口号
        self.port = conf.MILVUS_PORT
        # 设置 Milvus 数据库名称
        self.database = conf.MILVUS_DATABASE_NAME
        # 设置日志记录器
        self.logger = logger
        # 检查CUDA是否可用
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        # 日志提醒使用的是什么设备
        self.logger.info(f"使用设置：{self.device}")
        # 初始化重排序模型
        reranker_path = os.path.join(parent_dir, 'models', 'bge-reranker-large')
        self.reranker = CrossEncoder(reranker_path, device=self.device)
        # 初始化 BGE-M3 嵌入函数，使用 CPU 设备，不启用 FP16
        m3_path = os.path.join(parent_dir, 'models', 'bge-m3')
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=m3_path,
            use_fp16=(self.device == 'cuda'),
            device=self.device)
        # 获取稠密向量的维度# 1024
        self.dense_dim = self.embedding_function.dim["dense"]
        # 初始化 Milvus 客户端，连接到指定主机和数据库
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}", db_name=self.database)

        # 调用方法创建或加载 Milvus 集合
        self._create_or_load_collection()

    def _create_or_load_collection(self):
        if not self.client.has_collection(self.collection_name):
            # 创建集合 Schema，禁用自动 ID，启用动态字段
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 添加 ID 字段，作为主键，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            # 添加文本字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            # 添加稠密向量字段，FLOAT_VECTOR 类型，维度由嵌入函数指定
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT16_VECTOR, dim=self.dense_dim)
            # 添加稀疏向量字段，SPARSE_FLOAT_VECTOR 类型
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # 添加父块 ID 字段，VARCHAR 类型，最大长度 100
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=100)
            # 添加父块内容字段，VARCHAR 类型，最大长度 65535
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            # 添加学科类别字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)
            # 添加时间戳字段，VARCHAR 类型，最大长度 50
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128}
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_bulid":0.2}
            )
            self.client.create_collection(collection_name=self.collection_name,
                                          schema=schema,
                                          index_params=index_params)
            self.logger.info(f"集合 {self.collection_name} 创建成功")
        else:
            self.logger.info(f"集合 {self.collection_name} 已存在")
        self.client.load_collection(collection_name=self.collection_name)

    def add_docmuments(self, documents):
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_function(texts)
        data = []
        for i, doc in enumerate(documents):
            text_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            sparse_vector = {}
            row = embeddings["sparse"].getrow(i)
            indices = row.indices
            values = row.data
            for idx, value in zip(indices, values):
                sparse_vector[idx] = value
            data.append({
                "id": text_hash,
                "text": doc.page_content,
                "dense_vector": np.array(embeddings["dense"][i], dtype=np.float16),
                "sparse_vector": sparse_vector,
                "parent_id": doc.metadata.get("parent_id",""),
                "parent_content": doc.metadata.get("parent_content",""),
                "source": doc.metadata.get("source","unknown"),
                "timestamp": doc.metadata.get("timestamp","unknown")
            })
        if data:
            self.client.upsert(collection_name=self.collection_name, data=data)
            self.logger.info(f"已插入或更新{len(data)}个文档")

    def hybrid_search(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        """不带重排序的混合搜索"""
        logger.info(f"执行混合搜索，查询：'{query}'，k={k}，source_filter={source_filter}")
        query_embeddings = self.embedding_function([query])
        dense_query_vector = np.array(query_embeddings["dense"][0], dtype=np.float16)
        sparse_query_vector = {}
        row = query_embeddings["sparse"].getrow(0)
        indices = row.indices
        values = row.data
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value
        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        logger.info(f"过滤表达式：'{filter_expr}'")
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type":"IP", "params":{}},
            limit=k,
            expr=filter_expr
        )
        ranker = WeightedRanker(0.7, 1.0)
        search_results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )
        logger.info(f"Milvus hybrid_search 返回类型：{type(search_results)}")
        if not search_results or len(search_results) == 0:
            logger.warning("Milvus 返回空结果")
            return []
        results = search_results[0]
        logger.info(f"Milvus 返回的原始结果数量：{len(results)}")
        if len(results) == 0:
            logger.warning("Milvus 返回的结果列表为空")
            return []
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        logger.info(f"转换为子块数量：{len(sub_chunks)}")
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        logger.info(f"去重后父文档数量：{len(parent_docs)}")
        return parent_docs[:k]

    def hybrid_search_with_rerank(self, query, k=conf.RETRIEVAL_K, source_filter=None):
        query_embeddings = self.embedding_function([query])
        dense_query_vector = np.array(query_embeddings["dense"][0], dtype=np.float16)
        sparse_query_vector = {}
        row = query_embeddings["sparse"].getrow(0)
        indices = row.indices
        values = row.data
        for idx, value in zip(indices, values):
            sparse_query_vector[idx] = value
        filter_expr = f"source == '{source_filter}'" if source_filter else ""
        dense_request = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_query_vector],
            anns_field="sparse_vector",
            param={"metric_type":"IP", "params":{}},
            limit=k,
            expr=filter_expr
        )
        ranker = WeightedRanker(0.7, 1.0)
        search_results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"]
        )
        logger.info(f"Milvus hybrid_search_with_rerank 返回类型：{type(search_results)}")
        if not search_results or len(search_results) == 0:
            logger.warning("Milvus 返回空结果")
            return []
        results = search_results[0]
        logger.info(f"Milvus 返回的原始结果数量：{len(results)}")
        if len(results) == 0:
            logger.warning("Milvus 返回的结果列表为空")
            return []
        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in results]
        parent_docs = self._get_unique_parent_docs(sub_chunks)
        if len(parent_docs) < 2:
            return parent_docs[:conf.CANDIDATE_M]
        if parent_docs:
            pairs = [[query, doc.page_content] for doc in parent_docs]
            scores =self.reranker.predict(pairs)
            ranked_parent_docs = [doc for _, doc in sorted(zip(scores, parent_docs), reverse=True)]
        else:
            ranked_parent_docs = []
        return ranked_parent_docs[:conf.CANDIDATE_M]

    def _get_unique_parent_docs(self, sub_chunks):
        seen_parent_contents = set()
        unique_docs = []
        for chunk in sub_chunks:
            parent_content = chunk.metadata.get("parent_content", chunk.page_content)
            if parent_content and parent_content not in seen_parent_contents:
                unique_docs.append(Document(page_content=parent_content, metadata=chunk.metadata))
                seen_parent_contents.add(parent_content)
        return unique_docs

    def _doc_from_hit(self, hit):
        return Document(
            page_content=hit.get("text"),
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp")
            }
        )

if __name__ == '__main__':
    vector_store = VectorStore()

