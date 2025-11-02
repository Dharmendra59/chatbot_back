# ============================================================
# main.py — FastAPI Backend for Gemini Species Chatbot
# ============================================================
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from google import genai
import os, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from collections import deque
import uvicorn  # ✅ Import for running locally or Render

# ============================================================
# ⚙️ FastAPI App
# ============================================================
app = FastAPI(title="Gemini Animal Chatbot", version="1.0")

# ============================================================
# 🔑 Setup Gemini
# ============================================================
os.environ["GOOGLE_API_KEY"] = "AIzaSyCyTKqCfw7BeVvPRvYRLLtEZUhlpF-eC6A"
api_key = os.environ["GOOGLE_API_KEY"]
client = genai.Client(api_key=api_key)

# ============================================================
# 🧠 Knowledge Base
# ============================================================
KB = {
    "Horse": {"info": "Large herbivorous mammal used for riding, work and sport; lifespan ~25-30 years.", "health_precautions": "Regular hoof care, vaccinations, dental checks, balanced diet, deworming.", "emotion_signs": "Pinned ears, swishing tail, decreased appetite may indicate stress or pain."},
    "Frog": {"info": "Amphibian with permeable skin; many species rely on aquatic habitats; lifespans vary.", "health_precautions": "Keep water clean, avoid handling, maintain proper humidity and temp.", "emotion_signs": "Reduced movement or refusal to eat may indicate poor health or stress."},
    "Goat": {"info": "Domestic ruminant kept for milk, meat, and fiber; social animal.", "health_precautions": "Vaccinate, provide clean shelter, monitor hooves and parasites.", "emotion_signs": "Vocalization, head-butting or isolation can show distress or dominance."},
    "Housefly": {"info": "Common dipteran insect; short life cycle; thrives near organic waste.", "health_precautions": "Maintain sanitation, limit breeding sites, use screens and traps.", "emotion_signs": "Insects don't show human-like emotions; activity levels reflect environment."},
    "Monkey": {"info": "Primates with complex social structures; many species kept in sanctuaries, not as pets.", "health_precautions": "Do not keep as pets; zoonotic disease risks; proper enrichment needed.", "emotion_signs": "Facial expressions, vocalizations and social interactions indicate mood."},
    "Dog": {"info": "Domestic canid; social companion animal; lifespans ~10-15 years depending on breed.", "health_precautions": "Vaccinate, regular vet checks, parasite prevention, adequate exercise.", "emotion_signs": "Tail wagging, ear position, appetite, playfulness — indicators of mood."},
    "Cat": {"info": "Small carnivorous mammal; solitary but social; lifespans ~12-18 years.", "health_precautions": "Vaccinate, spay/neuter, keep litter clean, monitor weight and dental health.", "emotion_signs": "Purring, kneading indicate comfort; hiding or hissing indicate stress."},
    "Cow": {"info": "Large domestic bovine raised for milk and meat; social herd animal.", "health_precautions": "Clean bedding, mastitis prevention, vaccination and balanced diet.", "emotion_signs": "Changes in feed intake or social withdrawal signal problems."},
    "Buffalo": {"info": "Bovine similar to cows but adapted to wet environments.", "health_precautions": "Parasite control, clean water, foot care and vaccination.", "emotion_signs": "Alert posture and vocalization indicate discomfort or alarm."},
    "Mosquito": {"info": "Small dipteran insect; many species are vectors for disease.", "health_precautions": "Use nets, eliminate standing water, use repellents to reduce bites.", "emotion_signs": "No human-like emotions; population influenced by environment."},
    "Bee": {"info": "Flying insect essential for pollination; many species are social.", "health_precautions": "Avoid pesticides, provide forage, inspect hives for mites.", "emotion_signs": "Aggression when hive disturbed; otherwise focused on foraging."},
    "Peacock": {"info": "Large, colourful bird; male has ornate tail for display.", "health_precautions": "Provide shelter, monitor parasites, protect from predators.", "emotion_signs": "Tail display for mating; loud calls indicate alarm or territory."},
    "Crow": {"info": "Intelligent corvid; omnivorous and adaptable.", "health_precautions": "Avoid feeding junk food; protect habitats.", "emotion_signs": "Vocalization indicates curiosity or communication."},
    "Parrot": {"info": "Colorful psittacine bird; requires enrichment.", "health_precautions": "Social interaction, avoid seed-only diets, regular vet checks.", "emotion_signs": "Plucking, screaming or quietness indicate stress or boredom."},
    "Sparrow": {"info": "Small passerine bird common near humans.", "health_precautions": "Habitat preservation and clean water help.", "emotion_signs": "Active foraging shows health; reduced activity signals issues."},
    "Elephant": {"info": "Largest land mammal with complex social behavior.", "health_precautions": "Large space, social groups, foot care, proper nutrition.", "emotion_signs": "Ear flapping, trunk gestures convey emotional states."}
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
# 🚀 Entry Point for Render or Local
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Render sets PORT automatically
    uvicorn.run("main:app", host="0.0.0.0", port=port)

# ============================================================
# 🚀 Entry Point for Render
# ============================================================
else:
    uvicorn.run("main:app", host="0.0.0.0", port=5000)