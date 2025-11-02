# ============================================================
# main.py — FastAPI Backend for Gemini Species Chatbot
# ============================================================
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # ✅ Added CORS import
from google import genai
import os, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from collections import deque

# ============================================================
# ⚙ FastAPI App
# ============================================================
app = FastAPI(title="Gemini Animal Chatbot", version="1.0")

# ✅ CORS Middleware (must be placed immediately after app initialization)
origins = [
    "http://localhost:3000",  # local Next.js dev
    "https://https://superlative-maamoul-92da84.netlify.app/",  # change this to your deployed frontend domain
   ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] if you want to allow all origins (for testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🔑 Setup Gemini
# ============================================================
os.environ["GOOGLE_API_KEY"] = "AIzaSyBC2T4XHVl18MtC2t-HJCNA3H8y93zSPIU"
api_key = os.environ["GOOGLE_API_KEY"]
client = genai.Client(api_key=api_key)

# ============================================================
# 🧠 Knowledge Base
# ============================================================
KB = {
    "Horse": {"info": "Large herbivorous mammal used for riding, work and sport; lifespan ~25-30 years.",
              "health_precautions": "Regular hoof care, vaccinations, dental checks, balanced diet, deworming.",
              "emotion_signs": "Pinned ears, swishing tail, decreased appetite may indicate stress or pain."},
    "Frog": {"info": "Amphibian with permeable skin; many species rely on aquatic habitats; lifespans vary.",
             "health_precautions": "Keep water clean, avoid handling, maintain proper humidity and temp.",
             "emotion_signs": "Reduced movement or refusal to eat may indicate poor health or stress."},
    # ... (rest of KB unchanged)
}

# ============================================================
# 🧩 TF-IDF Retriever
# ============================================================
animal_texts = [f"{name}. {KB[name]['info']} {KB[name]['health_precautions']} {KB[name]['emotion_signs']}" for name in KB]
vectorizer = TfidfVectorizer().fit(animal_texts)
X = vectorizer.transform(animal_texts)
nn = NearestNeighbors(n_neighbors=3, metric='cosine').fit(X)

def retrieve_relevant_kb(user_query, k=3):
    qv = vectorizer.transform([user_query])
    dists, idxs = nn.kneighbors(qv, n_neighbors=k)
    results = []
    for idx in idxs[0]:
        name = list(KB.keys())[idx]
        results.append({"species": name, "text": animal_texts[idx]})
    return results

# ============================================================
# 💬 Chat Memory
# ============================================================
chat_history = deque(maxlen=10)

SYSTEM_PROMPT = (
    "You are a friendly animal expert chatbot. "
    "Answer questions about animals, their health, and emotions. "
    "Use previous conversation context to keep replies coherent.\n"
    "Be concise, natural, and kind."
)

def ask_gemini_context(user_message):
    context = retrieve_relevant_kb(user_message, k=3)
    history_text = ""
    if chat_history:
        history_text = "\nPrevious conversation:\n" + "\n".join(
            [f"User: {q}\nBot: {a}" for q, a in chat_history]
        )

    prompt = f"{SYSTEM_PROMPT}\n{history_text}\nContext: {json.dumps(context)}\nUser: {user_message}\nBot:"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt]
    )

    try:
        text = response.candidates[0].content.parts[0].text.strip()
    except Exception:
        text = str(response)

    chat_history.append((user_message, text))
    return text

# ============================================================
# 📩 Request Model
# ============================================================
class ChatRequest(BaseModel):
    message: str

# ============================================================
# 🧾 FastAPI Route
# ============================================================
@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        return JSONResponse({"error": "Empty message"}, status_code=400)
    
    reply = ask_gemini_context(req.message)
    return {"reply": reply}

# ============================================================
# 🚀 Local Development Entry Point
# ============================================================
if __name__ == "__main__":  # ✅ fixed typo
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)