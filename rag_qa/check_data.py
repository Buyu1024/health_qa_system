import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from core.vector_store import VectorStore
from base.config import Config

conf = Config()

def check_collection_data():
    """检查向量数据库中的数据"""
    vector_store = VectorStore()
    
    # 获取集合统计信息
    stats = vector_store.client.get_collection_stats(collection_name=conf.MILVUS_COLLECTION_NAME)
    print(f"\n{'='*60}")
    print(f"集合名称: {conf.MILVUS_COLLECTION_NAME}")
    print(f"集合统计信息: {stats}")
    print(f"{'='*60}\n")
    
    # 查询所有数据（限制数量）
    try:
        results = vector_store.client.query(
            collection_name=conf.MILVUS_COLLECTION_NAME,
            filter="",
            output_fields=["id", "text", "source"],
            limit=10
        )
        print(f"前10条数据:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. ID: {result.get('id', 'N/A')}")
            print(f"   Source: {result.get('source', 'N/A')}")
            text = result.get('text', 'N/A')
            print(f"   Text: {text[:100]}..." if len(text) > 100 else f"   Text: {text}")
        
        # 按 source 分组统计
        print(f"\n{'='*60}")
        print("按学科分类统计:")
        all_results = vector_store.client.query(
            collection_name=conf.MILVUS_COLLECTION_NAME,
            filter="",
            output_fields=["source"],
            limit=10000
        )
        source_count = {}
        for result in all_results:
            source = result.get('source', 'unknown')
            source_count[source] = source_count.get(source, 0) + 1
        
        for source, count in sorted(source_count.items()):
            print(f"  {source}: {count} 条")
        print(f"  总计: {len(all_results)} 条")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"查询出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_collection_data()
