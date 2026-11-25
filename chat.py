import os
from zhipuai import ZhipuAI

from dotenv import load_dotenv

load_dotenv()
_api_key = os.environ.get("ZHIPUAI_API_KEY", "").strip()
if not _api_key:
    raise SystemExit("未配置 ZHIPUAI_API_KEY：请复制 .env.example 为 .env 并填入你的 Key")
client = ZhipuAI(api_key=_api_key)

reply = client.chat.completions.create(
    model="glm-4-flash",   # 修改为正确的免费模型名称
    messages=[
        {"role": "user", "content": "用一句话解释什么是注意力机制"}
    ],
)
print(reply.choices[0].message.content)

