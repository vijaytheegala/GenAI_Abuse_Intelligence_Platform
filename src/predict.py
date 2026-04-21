import pickle

# Load once
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_text(text: str, threshold: float = 0.4):
    text_vec = vectorizer.transform([text])
    prob = model.predict_proba(text_vec)[0][1]
    prediction = int(prob > threshold)

    return {
        "text": text,
        "harmful": prediction,
        "confidence": float(prob)
    }
