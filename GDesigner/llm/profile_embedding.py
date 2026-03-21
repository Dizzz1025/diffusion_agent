from sentence_transformers import SentenceTransformer

MODEL_PATH = '/home/zhangdi24/diffusion_agent/all-MiniLM-L6-v2'
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_PATH, device="cpu")
    return _embedding_model

def get_sentence_embedding(sentence):
    model = get_embedding_model()
    embedding = model.encode(sentence, convert_to_numpy=True)
    return embedding
