import jieba

from base.logger import logger

def preprocess_text(text):
    logger.info("开始预处理文本")
    try:
        return jieba.lcut(text.lower())
    except Exception as e:
        logger.error(f"文本预处理失败 {e}")
        return []


if __name__ == '__main__':
    text = "这是一个测试文本aasASAs"
    print(preprocess_text(text))