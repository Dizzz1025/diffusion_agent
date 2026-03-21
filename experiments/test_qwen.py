from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1",
)

models = client.models.list()
print(models)

model_name = models.data[0].id

resp = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)

# from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]
# print('sentences')
# model = SentenceTransformer('/home/zhangdi24/diffusion_agent/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
# print(embeddings)
