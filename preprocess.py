import os
import json
from bs4 import BeautifulSoup
import re

def extract_text_and_title_from_html(html_filepath):
    """
    从指定的 HTML 文件中提取标题和正文文本。

    Args:
        html_filepath (str): HTML 文件的路径。

    Returns:
        tuple: (标题, 正文文本) 或 (None, None) 如果失败。
    """
    try:
        with open(html_filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'lxml') # 或者使用 'html.parser'

        # --- 提取标题 ---
        title_tag = soup.find('title')
        title_string = title_tag.string if title_tag else None
        # 确保 title_string 不为 None 才调用 strip()
        title = title_string.strip() if title_string else os.path.basename(html_filepath)
        title = title.replace('.html', '') # 清理标题

        # --- 定位正文内容 ---
        # 根据之前的讨论，优先查找 <content> 或特定 class
        content_tag = soup.find('content')
        if not content_tag:
            content_tag = soup.find('div', class_='rich_media_content') # 微信文章常见
        if not content_tag:
            content_tag = soup.find('article') # HTML5 语义标签
        if not content_tag:
            content_tag = soup.find('main') # HTML5 语义标签
        if not content_tag:
             content_tag = soup.find('body') # 最后尝试 body

        if content_tag:
            # 移除不需要的元素
            for element in content_tag.find_all(['script', 'style', 'iframe', 'img', 'video', 'audio', 'canvas']):
                element.decompose()
            
            # 移除特定的广告和无关内容
            for element in content_tag.find_all(class_=['rich_media_tool', 'rich_media_meta_list', 'rich_media_meta_nickname', 
                                                      'rich_media_meta_text', 'rich_media_meta_link', 'rich_media_meta_primary']):
                element.decompose()
            
            # 移除更多广告和无关内容
            ad_patterns = [
                r'阅读原文', r'广告', r'点击查看', r'扫码关注', r'长按识别', r'关注我们',
                r'关注公众号', r'关注订阅号', r'关注服务号', r'关注企业号', r'关注小程序',
                r'点击阅读', r'点击进入', r'点击链接', r'点击图片', r'点击视频',
                r'欢迎关注', r'欢迎订阅', r'欢迎分享', r'欢迎转发', r'欢迎点赞',
                r'欢迎评论', r'欢迎收藏', r'欢迎打赏', r'欢迎赞赏', r'欢迎支持',
                r'欢迎加入', r'欢迎参与', r'欢迎投稿', r'欢迎合作', r'欢迎咨询',
                r'欢迎联系', r'欢迎交流', r'欢迎讨论', r'欢迎反馈', r'欢迎建议',
                r'欢迎举报', r'欢迎投诉', 
                r'活动', r'报名', r'讲座', r'会议', r'论坛', r'沙龙',
                r'时间', r'地点', r'费用', r'主办', r'承办', r'协办',
                r'嘉宾', r'议程', r'咨询', r'联系方式', r'二维码', r'详情',
                r'预约', r'报名链接', r'扫码', r'免费', r'收费', r'限时',
                r'名额有限', r'报名截止', r'主办方', r'承办方', r'协办方'
            ]
            
            # 移除包含广告文本的元素
            for pattern in ad_patterns:
                for element in content_tag.find_all(text=re.compile(pattern)):
                    if element.parent:
                        element.parent.decompose()
            
            # 获取文本，保留段落结构
            text = content_tag.get_text(separator='\n', strip=True)
            
            # 清理文本
            # 1. 移除多余的空行
            text = re.sub(r'\n\s*\n', '\n', text)
            # 2. 移除特定的广告和宣传文本
            for pattern in ad_patterns:
                # 移除包含关键词的整行，或者只移除关键词本身，这里尝试移除整行以提高清理效果
                text = re.sub(f'.*{pattern}.*$', '', text, flags=re.MULTILINE)
            # 3. 移除URL
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            # 4. 移除特殊字符
            text = re.sub(r'[^\w\s\u4e00-\u9fff：；""''（）【】《》、]', '', text)
            # 5. 移除重复的标点符号
            text = re.sub(r'[。，！？：；]{2,}', lambda m: m.group()[0], text)
            # 6. 移除空白字符
            text = re.sub(r'\s+', ' ', text)
            # 7. 移除纯数字行
            text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)

            # 8. 检查是否包含活动宣传等模式，如果包含则舍弃整篇文本
            is_promotional = False
            for pattern in ad_patterns:
                if re.search(pattern, text):
                    is_promotional = True
                    break

            if is_promotional:
                print(f"    检测到活动宣传/广告内容，舍弃文件 {html_filepath} 的文本。")
                return title, None # 舍弃文本内容
            else:
                return title, text.strip() # 返回清理后的文本
        else:
            print(f"警告：在文件 {html_filepath} 中未找到明确的正文标签。")
            return title, None # 返回标题，但文本为 None

    except FileNotFoundError:
        print(f"错误：文件 {html_filepath} 未找到。")
        return None, None
    except Exception as e:
        print(f"处理文件 {html_filepath} 时出错: {e}")
        return None, None

def split_text(text, chunk_size=500, chunk_overlap=50, mode="sentence"):
    """
    优化的文本分割函数
    
    Args:
        text: 输入文本
        chunk_size: 目标块大小
        chunk_overlap: 重叠大小
        mode: 分割模式 ("char"|"sentence")
    
    Returns:
        文本块列表
    """
    if not text or chunk_size <= chunk_overlap:
        return []
    
    text = text.strip()
    if not text:
        return []
    
    if mode == "sentence":
        return _split_by_sentence(text, chunk_size, chunk_overlap)
    else:
        return _split_by_char(text, chunk_size, chunk_overlap)

def _split_by_char(text, chunk_size, chunk_overlap):
    chunks = []
    text_length = len(text)
    step = chunk_size - chunk_overlap
    
    for i in range(0, text_length, step):
        chunk = text[i:i+chunk_size].strip()
        if chunk and (not chunks or chunk != chunks[-1]):  # 避免重复
            chunks.append(chunk)
    return chunks

def _split_by_sentence(text, chunk_size, chunk_overlap):
    sentences = re.split(r'(?<=[。！？])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sent_len = len(sentence)
        if current_length + sent_len <= chunk_size:
            current_chunk.append(sentence)
            current_length += sent_len
        else:
            if current_chunk:
                chunk = ' '.join(current_chunk).strip()
                if chunk:
                    chunks.append(chunk)
            
            # 计算重叠部分
            overlap = []
            overlap_len = 0
            for sent in reversed(current_chunk):
                if overlap_len + len(sent) > chunk_overlap:
                    break
                overlap.insert(0, sent)
                overlap_len += len(sent)
            
            current_chunk = overlap + [sentence]
            current_length = overlap_len + sent_len
    
    if current_chunk:
        chunk = ' '.join(current_chunk).strip()
        if chunk:
            chunks.append(chunk)
    
    return chunks

# --- 配置 ---
html_directory = './data/html' # **** 修改为你的 HTML 文件夹路径 ****
output_json_path = './data/processed_datanew.json' # **** 输出 JSON 文件路径 ****
CHUNK_SIZE = 512  # 每个文本块的目标大小（字符数）
CHUNK_OVERLAP = 50 # 相邻文本块的重叠大小（字符数）

# --- 主处理逻辑 ---
all_data_for_milvus = []
file_count = 0
chunk_count = 0

print(f"开始处理目录 '{html_directory}' 中的 HTML 文件...")

# 确保输出目录存在
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

html_files = [f for f in os.listdir(html_directory) if f.endswith('.html')]
print(f"找到 {len(html_files)} 个 HTML 文件。")

for filename in html_files:
    filepath = os.path.join(html_directory, filename)
    print(f"  处理文件: {filename} ...")
    file_count += 1

    title, main_text = extract_text_and_title_from_html(filepath)

    if main_text:
        chunks = split_text(main_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"    提取到文本，分割成 {len(chunks)} 个块。")

        for i, chunk in enumerate(chunks):
            chunk_count += 1
            # 构建符合 milvus_utils.py 期望的字典结构
            milvus_entry = {
                "id": f"{filename}_{i}", # 创建一个唯一的 ID (文件名 + 块索引)
                "title": title or filename, # 使用提取的标题或文件名
                "abstract": chunk, # 将文本块放入 'abstract' 字段
                "source_file": filename, # 添加原始文件名以供参考
                "chunk_index": i
            }
            all_data_for_milvus.append(milvus_entry)
    else:
        print(f"    警告：未能从 {filename} 提取有效文本内容。")

print(f"\n处理完成。共处理 {file_count} 个文件，生成 {chunk_count} 个文本块。")

# --- 保存为 JSON ---
try:
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data_for_milvus, f, ensure_ascii=False, indent=4)
    print(f"结果已保存到: {output_json_path}")
except Exception as e:
    print(f"错误：无法写入 JSON 文件 {output_json_path}: {e}")
