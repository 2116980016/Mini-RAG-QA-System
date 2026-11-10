import gradio as gr
import os
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ========== 1. 配置你的 API Key ==========
# ⚠️ 请在这里填入你的智谱 API Key！
os.environ["ZHIPUAI_API_KEY"] = "9abc27d7a70449be8b9af53eaa58a12a.GaQVnPRlpvsIjugV"

print("正在加载本地 FAISS 向量库，请稍等...")
# 直接加载之前跑通存下来的 faiss_index，不用重新花钱算向量
embeddings = ZhipuAIEmbeddings(model="embedding-3")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ========== 2. 配置大模型和防幻觉 Prompt ==========
llm = ChatOpenAI(
    model="glm-4-flash",
    openai_api_key=os.environ["ZHIPUAI_API_KEY"],
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1
)

template = """你是一个严谨的文档问答助手。请仅根据以下检索到的背景知识回答问题。
如果你在背景知识中找不到答案，请直接回答“根据现有资料无法回答”。
请务必在回答后标明引用的片段编号，例如 [片段1]。

【背景知识】：
{context}

【用户问题】：
{question}
"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([f"[片段{i+1}]: {doc.page_content}" for i, doc in enumerate(docs)])

# ========== 3. 组装 RAG 链 ==========
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ========== 4. 定义网页聊天逻辑 ==========
def chat_with_rag(message, history):
    if not message.strip():
        return "请输入有效问题。"
    try:
        # 调用 RAG 链生成回答
        response = rag_chain.invoke(message)
        return response
    except Exception as e:
        return f"系统报错：{str(e)}"

# ========== 5. 启动 Gradio 网页 ==========
print("正在启动网页界面...")
demo = gr.ChatInterface(
    fn=chat_with_rag,
    title="📚 《红楼梦》RAG 专属问答系统",
    description="基于 LangChain + FAISS + GLM-4。所有回答严格依据原著，并附带引用来源。",
    examples=[
        "贾宝玉神游太虚幻境看到了什么？",
        "贾宝玉最后娶了谁？",
        "请问苹果公司的CEO是谁？(测试拒答)"
    ]
)

if __name__ == "__main__":
    demo.launch()