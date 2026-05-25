import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from datetime import datetime

from base.config import Config
from base.logger import logger
from rag_qa.document_loaders import OCRDOCLoader,OCRPDFLoader,OCRPPTLoader,OCRIMGLoader
from text_spliter.chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
from text_spliter.model_text_spliter import AliTextSplitter

conf = Config()
document_loaders = {
    ".txt": TextLoader,
    ".pdf": OCRPDFLoader,
    ".docx": OCRDOCLoader,
    ".ppt": OCRPPTLoader,
    ".pptx": OCRPPTLoader,
    ".png": OCRIMGLoader,
    ".jpg": OCRIMGLoader,
    ".md": UnstructuredMarkdownLoader
}

def load_documents_from_directory(directory_path):
    documents = []
    supported_extensions = document_loaders.keys()
    for root, _, files in os.walk(directory_path):
        print(root)
        source = os.path.basename(root)
        print(source)
        print(files)
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            print("-"*70)
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in supported_extensions:
                try:
                    loader_class = document_loaders[file_extension]
                    if file_extension == ".txt":
                        loader = loader_class(file_path,encoding="utf-8")
                    else:
                        loader = loader_class(file_path)
                    loaded_docs = loader.load()
                    # print(loaded_docs)
                    for doc in loaded_docs:
                        doc.metadata["source"] = source
                        doc.metadata["file_path"] = file_path
                        doc.metadata["timestamp"] = datetime.now().isoformat()
                    documents.extend(loaded_docs)
                    logger.info(f"成功加载文件：{file_path}")
                except Exception as e:
                    logger.error(f"加载文件{file_path}失败：{str(e)}")
            else:
                logger.warning(f"不支持的文件类型：{file_path}")
    return documents

def process_documents(directory_path,
                      parent_chunk_size= conf.PARENT_CHUNK_SIZE,
                      child_chunk_size= conf.CHILD_CHUNK_SIZE,
                      chunk_overlap= conf.CHUNK_OVERLAP,
                      model_pattern= False):
    documents = load_documents_from_directory(directory_path)
    logger.info(f"加载的文档数量：{len(documents)}")
    # 文本切分模式
    if model_pattern:
        parent_splitter = AliTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
        child_splitter = AliTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    else:
        parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
        child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    child_chunks = []
    for i, doc in enumerate(documents):
        file_extension = os.path.splitext(doc.metadata["file_path"])[1].lower()
        is_md = (file_extension == ".md")
        parent_splitter_to_use = markdown_parent_splitter if is_md else parent_splitter
        child_splitter_to_use = markdown_child_splitter if is_md else child_splitter
        logger.info(f"正在处理文档：{doc.metadata['file_path']},使用的切分器：{'Markdown' if is_md else ('AliTextSplitter' if model_pattern else 'ChineseRecursive')}")
        parent_docs = parent_splitter_to_use.split_documents([doc])
        for j, parent_doc in enumerate(parent_docs):
            parent_id = f"doc_{i}_parent_{j}"
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])
            for k, sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata["parent_id"] = parent_id
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                sub_chunk.metadata["id"] = f"{parent_id}_child_{k}"
                child_chunks.append(sub_chunk)
    logger.info(f"切分后的子块数量：{len(child_chunks)}")
    return child_chunks


if __name__ == '__main__':
    # load_documents_from_directory(r"..\data")
    process_documents(r"..\data")