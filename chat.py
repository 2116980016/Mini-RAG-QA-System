import os
from zhipuai import ZhipuAI

# 建议：在终端运行 export ZHIPUAI_API_KEY="你的新key" 或者使用 python-dotenv 库
client = ZhipuAI(api_key="你的API")

reply = client.chat.completions.create(
    model="glm-4-flash",   # 修改为正确的免费模型名称
    messages=[
        {"role": "user", "content": "用一句话解释什么是注意力机制"}
    ],
)
print(reply.choices[0].message.content)

