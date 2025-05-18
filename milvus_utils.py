import streamlit as st
# Use MilvusClient for Lite version
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
import time
import os

# Import config variables including the global map
from config import (
    MILVUS_LITE_DATA_PATH, COLLECTION_NAME, EMBEDDING_DIM,
    MAX_ARTICLES_TO_INDEX, INDEX_METRIC_TYPE, INDEX_TYPE, INDEX_PARAMS,
    SEARCH_PARAMS, TOP_K, id_to_doc_map
)

@st.cache_resource
def get_milvus_client():
    """Initializes and returns a MilvusClient instance for Milvus Lite."""
    try:
        st.write(f"Initializing Milvus Lite client with data path: {MILVUS_LITE_DATA_PATH}")
        # Ensure the directory for the data file exists
        os.makedirs(os.path.dirname(MILVUS_LITE_DATA_PATH), exist_ok=True)
        # The client connects to the local file specified
        client = MilvusClient(uri=MILVUS_LITE_DATA_PATH)
        st.success("Milvus Lite client initialized!")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Milvus Lite client: {e}")
        return None

@st.cache_resource
def setup_milvus_collection(_client):
    """Ensures the specified collection exists and is set up correctly in Milvus Lite."""
    if not _client:
        st.error("Milvus client not available.")
        return False
    try:
        collection_name = COLLECTION_NAME
        dim = EMBEDDING_DIM

        has_collection = collection_name in _client.list_collections()

        if not has_collection:
            st.write(f"Collection '{collection_name}' not found. Creating...")
            # Define fields using new API style if needed (older style might still work)
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                # You can add other scalar fields directly here for storage
                FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=500), # Example
            ]
            schema = CollectionSchema(fields, f"PubMed Lite RAG (dim={dim})")

            _client.create_collection(
                collection_name=collection_name,
                schema=schema # Pass schema directly or define dimension/primary field name
                # Or simpler:
                # dimension=dim,
                # primary_field_name="id",
                # vector_field_name="embedding",
                # metric_type=INDEX_METRIC_TYPE
            )
            st.write(f"Collection '{collection_name}' created.")

            # Create an index
            st.write(f"Creating index ({INDEX_TYPE})...")
            index_params = _client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type=INDEX_TYPE,
                metric_type=INDEX_METRIC_TYPE,
                params=INDEX_PARAMS
            )
            _client.create_index(collection_name, index_params)
            st.success(f"Index created for collection '{collection_name}'.")
        else:
            st.write(f"Found existing collection: '{collection_name}'.")
            # Optional: Check schema compatibility if needed

        # Determine current entity count (fallback between num_entities and stats)
        try:
            if hasattr(_client, 'num_entities'):
                current_count = _client.num_entities(collection_name)
            else:
                stats = _client.get_collection_stats(collection_name)
                current_count = int(stats.get("row_count", stats.get("rowCount", 0)))
            st.write(f"Collection '{collection_name}' ready. Current entity count: {current_count}")
        except Exception:
            st.write(f"Collection '{collection_name}' ready.")

        return True # Indicate collection is ready

    except Exception as e:
        st.error(f"Error setting up Milvus collection '{COLLECTION_NAME}': {e}")
        return False


def index_data_if_needed(client, data, embedding_model):
    """Checks if data needs indexing and performs it using MilvusClient."""
    global id_to_doc_map # Modify the global map

    if not client:
        st.error("Milvus client not available for indexing.")
        return False

    collection_name = COLLECTION_NAME
    # Retrieve current entity count with fallback
    try:
        if hasattr(client, 'num_entities'):
            current_count = client.num_entities(collection_name)
        else:
            stats = client.get_collection_stats(collection_name)
            current_count = int(stats.get("row_count", stats.get("rowCount", 0)))
    except Exception:
        st.write(f"Could not retrieve entity count, attempting to (re)setup collection.")
        if not setup_milvus_collection(client):
            return False
        current_count = 0  # Assume empty after setup

    st.write(f"Entities currently in Milvus collection '{collection_name}': {current_count}")

    data_to_index = data[:MAX_ARTICLES_TO_INDEX] # Limit data for demo
    needed_count = 0
    docs_for_embedding = []
    data_to_insert = [] # List of dictionaries for MilvusClient insert
    temp_id_map = {} # Build a temporary map first

    # Prepare data
    with st.spinner("Preparing data for indexing..."):
        for i, doc in enumerate(data_to_index):
             title = doc.get('title', '') or ""
             abstract = doc.get('abstract', '') or ""
             content = f"Title: {title}\nAbstract: {abstract}".strip()
             if not content:
                 continue

             doc_id = i # Use list index as ID
             needed_count += 1
             temp_id_map[doc_id] = {
                 'title': title, 'abstract': abstract, 'content': content
             }
             docs_for_embedding.append(content)
             # Prepare data in dict format for MilvusClient
             data_to_insert.append({
                 "id": doc_id,
                 "embedding": None, # Placeholder, will be filled after encoding
                 "content_preview": content[:500] # Store preview if field exists
             })


    if current_count < needed_count and docs_for_embedding:
        st.warning(f"Indexing required ({current_count}/{needed_count} documents found). This may take a while...")

        st.write(f"Embedding {len(docs_for_embedding)} documents...")
        with st.spinner("Generating embeddings..."):
            start_embed = time.time()
            embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
            end_embed = time.time()
            st.write(f"Embedding took {end_embed - start_embed:.2f} seconds.")

        # Fill in the embeddings
        for i, emb in enumerate(embeddings):
            data_to_insert[i]["embedding"] = emb

        st.write("Inserting data into Milvus Lite...")
        with st.spinner("Inserting..."):
            try:
                start_insert = time.time()
                # MilvusClient uses insert() with list of dicts
                res = client.insert(collection_name=collection_name, data=data_to_insert)
                # Milvus Lite might automatically flush or sync, explicit flush isn't usually needed/available
                end_insert = time.time()
                # 使用 len(data_to_insert) 作为成功插入的数量，因为 res 可能没有 primary_keys 属性
                inserted_count = len(data_to_insert)
                st.success(f"Successfully attempted to index {inserted_count} documents. Insert took {end_insert - start_insert:.2f} seconds.")
                # Update the global map ONLY after successful insertion attempt
                id_to_doc_map.update(temp_id_map)
                return True
            except Exception as e:
                st.error(f"Error inserting data into Milvus Lite: {e}")
                return False
    elif current_count >= needed_count:
        st.write("Data count suggests indexing is complete.")
        # Populate the global map if it's empty but indexing isn't needed
        if not id_to_doc_map:
            id_to_doc_map.update(temp_id_map)
        return True
    else: # No docs_for_embedding found
         st.error("No valid text content found in the data to index.")
         return False

# def index_data_if_needed(client, data, embedding_model):
#     """Checks if data needs indexing and performs it using MilvusClient."""
#     global id_to_doc_map  # Modify the global map
#     id_to_doc_map.clear()  # 清空旧映射

#     if not client:
#         st.error("Milvus client not available for indexing.")
#         return False

#     collection_name = COLLECTION_NAME

#     # Retrieve current entity count with fallback
#     try:
#         if hasattr(client, 'num_entities'):
#             current_count = client.num_entities(collection_name)
#         else:
#             stats = client.get_collection_stats(collection_name)
#             current_count = int(stats.get("row_count", stats.get("rowCount", 0)))
#     except Exception:
#         st.warning(f"Could not retrieve entity count, attempting to (re)setup collection.")
#         if not setup_milvus_collection(client):
#             return False
#         current_count = 0  # Assume empty after setup

#     st.info(f"Entities currently in Milvus collection '{collection_name}': {current_count}")

#     data_to_index = data[:MAX_ARTICLES_TO_INDEX]  # Limit data for demo
#     needed_count = len(data_to_index)
#     docs_for_embedding = []
#     data_to_insert = []  # List of dictionaries for MilvusClient insert
#     temp_id_map = {}  # Build a temporary map first

#     start_id = current_count
#     inserted_ids = []

#     for i, doc in enumerate(data_to_index):
#         doc_id = start_id + i + 1
#         text = doc["abstract"] or doc["title"] or "Untitled"
#         try:
#             embedding = embedding_model.encode(text)
#         except Exception as e:
#             st.warning(f"生成嵌入失败，跳过文档 ID: {doc_id}，错误: {e}")
#             continue

#         try:
#             client.insert(collection_name, data=[doc_id, embedding])
#             inserted_ids.append(doc_id)
#             temp_id_map[doc_id] = doc
#         except Exception as e:
#             st.warning(f"插入 Milvus 失败，文档 ID: {doc_id}，错误: {e}")
#             for field_name, value in data[0].items():
#                 print(f"{field_name}: {type(value)}")
#             continue
#         # Merge only successfully inserted IDs
#     for doc_id in inserted_ids:
#         id_to_doc_map[doc_id] = temp_id_map[doc_id]

#     st.success(f"成功索引 {len(inserted_ids)} 条数据。")
#     return True


def search_similar_documents(client, query, embedding_model):
    """Searches Milvus Lite for documents similar to the query using MilvusClient."""
    if not client or not embedding_model:
        st.error("Milvus client or embedding model not available for search.")
        return [], []

    collection_name = COLLECTION_NAME
    try:
        query_embedding = embedding_model.encode([query])[0]

        # 重写search调用，使用更兼容的方式
        search_params = {
            "collection_name": collection_name,
            "data": [query_embedding],
            "anns_field": "embedding",
            "limit": TOP_K,
            "output_fields": ["id"]
        }
        
        # 尝试不同的方式传递搜索参数
        if hasattr(client, 'search_with_params'):
            # 如果存在专门的方法
            res = client.search_with_params(**search_params, search_params=SEARCH_PARAMS)
        else:
            # 标准方法，直接设置参数（当前版本会导致参数冲突）
            try:
                # 尝试1：不传递param参数
                res = client.search(**search_params)
            except Exception as e1:
                st.warning(f"搜索尝试1失败: {e1}，将尝试备用方法...")
                try:
                    # 尝试2：通过搜索参数关键字传递
                    res = client.search(**search_params, **SEARCH_PARAMS)
                except Exception as e2:
                    st.warning(f"搜索尝试2失败: {e2}，将尝试最后一种方法...")
                    # 尝试3：结合参数
                    final_params = search_params.copy()
                    final_params["nprobe"] = SEARCH_PARAMS.get("nprobe", 16)
                    res = client.search(**final_params)

        # Process results (structure might differ slightly)
        # client.search returns a list of lists of hits (one list per query vector)
        if not res or not res[0]:
            return [], []

        hit_ids = [hit['id'] for hit in res[0]]
        distances = [hit['distance'] for hit in res[0]]
        return hit_ids, distances
    except Exception as e:
        st.error(f"Error during Milvus Lite search: {e}")
        return [], [] 
    
from scipy.sparse import csr_matrix

def search_similar_documents2(client, query, embedding_model):
    if not client or not embedding_model:
        st.error("Milvus client or embedding model not available for search.")
        return [], []

    collection_name = COLLECTION_NAME
    try:
        # 生成嵌入向量
        embeddings = embedding_model.encode(query, convert_to_tensor=False)

        # 提取密集向量并确保格式
        dense_vector = embeddings['dense_vecs'].reshape(1, -1)

        # 提取稀疏向量并确保是 CSR 格式
        sparse_vector = embeddings['sparse_vecs']
        if not isinstance(sparse_vector, csr_matrix):
            sparse_vector = csr_matrix(sparse_vector)

        if sparse_vector.shape[0] != 1:
            sparse_vector = sparse_vector.reshape(1, -1)

        # 构建混合搜索请求
        reqs = [
            AnnSearchRequest(
                data=dense_vector,
                anns_field="dense_vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=TOP_K
            ),
            AnnSearchRequest(
                data=sparse_vector,
                anns_field="sparse_vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=TOP_K
            )
        ]

        # 执行混合搜索
        ranker = RRFRanker(k=60)
        search_results = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=ranker,
            output_fields=["id", "text"]
        )

        # 处理结果
        if not search_results or not search_results[0]:
            return [], []

        hit_ids = [hit['id'] for hit in search_results[0]]
        distances = [hit['distance'] for hit in search_results[0]]
        return hit_ids, distances

    except Exception as e:
        st.error(f"Error during Milvus Lite hybrid search: {e}")
        return [], []