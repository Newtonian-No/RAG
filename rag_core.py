import streamlit as st
import torch
from config import MAX_NEW_TOKENS_GEN, TEMPERATURE, TOP_P, REPETITION_PENALTY

def generate_answer(query, context_docs, gen_model, tokenizer, conversation_history=None):
    """Generates an answer using the LLM based on query and context."""
    if not context_docs:
        return "I couldn't find relevant documents to answer your question."
    if not gen_model or not tokenizer:
         st.error("Generation model or tokenizer not available.")
         return "Error: Generation components not loaded."

    context = "\n\n---\n\n".join([doc['content'] for doc in context_docs]) # Combine retrieved docs
    if conversation_history:
        history_str = "\n\n".join([f"**用户问题:** {history['query']}\n**系统回答:** {history['answer']}" for history in conversation_history])
        prompt = f"""你是一个医学问答助手。请根据以下对话历史和新问题，在已索引的医疗文档中检索相关信息并生成答案。

对话历史：
{history_str}

新问题：
{query}

上下文：
{context}

要求：
- 使用中文回答；
- 回答应简洁、准确；
- 如果上下文中没有相关信息，请说明无法回答。
- **若上下文有相关信息**：
  - 使用“证据等级”标注（如“强证据”“弱证据”）；
  - 分点列出支持依据。
- **若上下文无相关信息**：
  - 提供可能的替代搜索关键词（如“建议查询：XXX相关研究”）；
  - 说明当前数据集的时间范围（如“本系统数据更新至2023年”）。
- **语言要求**：避免绝对化表述（如“一定”“必须”），改用“可能”“建议”。
- 参考对话历史中提到的相关信息，并提供证据支持。
"""
    else:
        prompt = f"""你是一个医学问答助手。请根据以下上下文内容，回答用户的问题：

上下文：
{context}

问题：
{query}

要求：
- 使用中文回答；
- 回答应简洁、准确；
- 如果上下文中没有相关信息，请说明无法回答。
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(gen_model.device)
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_GEN,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id # Important for open-end generation
            )
        # Decode only the newly generated tokens, excluding the prompt
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return "Sorry, I encountered an error while generating the answer." 
    

def generate_answer_openai(query, retrieved_docs, client):
    context = "\n\n".join([doc["abstract"] for doc in retrieved_docs])
    prompt = f"""
你是一个医学问答助手。请根据以下上下文内容，回答用户的问题：

上下文：
{context}

问题：
{query}

要求：
- 使用中文回答；
- 回答应简洁、准确；
- 如果上下文中没有相关信息，请说明无法回答。
- **若上下文有相关信息**：
  - 使用“证据等级”标注（如“强证据”“弱证据”）；
  - 分点列出支持依据。
- **若上下文无相关信息**：
  - 提供可能的替代搜索关键词（如“建议查询：XXX相关研究”）；
  - 说明当前数据集的时间范围（如“本系统数据更新至2023年”）。
- **语言要求**：避免绝对化表述（如“一定”“必须”），改用“可能”“建议”。
"""

    try:
        response = client.chat.completions.create(
            model="qwen-max",  # 或 qwen-plus / qwen-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"生成答案时出错: {e}")
        return "无法生成答案，请稍后再试。"