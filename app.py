import os
import string
import markdown
import requests
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
from gradio_client import Client

# LangChain imports for generic QA and routing.
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, ChatGeneration

# -----------------------
# Load environment variables
load_dotenv()

# -----------------------
# Flask setup
app = Flask(__name__)

# -----------------------
# MongoDB setup for disease info
mongodb_password = os.getenv("MONGODB_PASSWORD")
client_mongo = MongoClient(
    f"mongodb+srv://samratray:{mongodb_password}"
    "@cluster0.fcgztso.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
db = client_mongo["medichat"]
collection = db["disease_info"]

def retrieve_info_from_mongo(disease_name):
    doc = collection.find_one(
        {"Disease": {"$regex": f"^{disease_name}$", "$options": "i"}}
    )
    if doc:
        return {"disease": disease_name, "description": doc["Description"]}
    return {"disease": None, "description": "Information not found for the given disease."}

# -----------------------
# Helper function for text cleaning
def clean_text(text: str) -> str:
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return " ".join(text.translate(translator).split())

# -----------------------
# Create Gradio Clients for your Spaces
# These clients connect to your deployed Gradio Spaces.
symptom_client = Client("samratray/my_bert_space1")
disease_client = Client("samratray/my_bert_space2")

# -----------------------
# Functions to call Gradio Spaces APIs

def classify_symptoms_api(text: str, threshold: float = 0.8):
    """
    Calls the symptom classification API.
    Expected output formats may include:
      - A dict with keys "label" and "confidences"
      - A dict of {symptom: confidence, ...}
      - A list of dicts with "label" and "confidence"
      - A list of symptom strings.
    """
    result = symptom_client.predict(
        text=text,
        threshold=threshold,
        api_name="/predict"
    )
    return result

def predict_disease_api(detected: list, threshold: float = 0.995):
    """
    Calls the disease prediction API.
    Passes symptom labels as a comma-separated string.
    Expected output is either a dictionary mapping disease names to probabilities
    or a dict with keys like "label" and "confidences".
    """
    detected_string = ", ".join(detected)
    result = disease_client.predict(
        detected=detected_string,
        threshold=threshold,
        output_format="dict",
        api_name="/predict"
    )
    return result

# -----------------------
# Gemma2LLM for generic QA and routing decisions (fallback for non-medical queries)
class Gemma2LLM(BaseLLM):
    @property
    def _llm_type(self) -> str:
        return "gemma2"

    def _generate(
        self, prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None
    ) -> LLMResult:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = "https://api.groq.com/openai/v1/chat/completions"
        gens = []
        for prompt in prompts:
            payload = {"model": "gemma2-9b-it", "messages": [{"role": "user", "content": prompt}]}
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            gens.append(ChatGeneration(message=AIMessage(content=content), generation_info={}))
        return LLMResult(generations=[[g] for g in gens])

llm_for_qa = Gemma2LLM()

# -----------------------
# FAISS and RetrievalQA setup for general queries
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_dir = "faiss_index"
if os.path.exists(index_dir) and os.listdir(index_dir):
    vector_store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
else:
    raise RuntimeError("FAISS index not found; build it first")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
prompt_template = (
    "Use the following context to answer the question in markdown:\n"
    "(Only answer if the Question is Medical Related or else respond with 'I am a Medical Chatbot and am not specialized in other domains. Please stick with my specialized domain') \n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)
QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_for_qa,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# -----------------------
# LLM-based query routing function
def route_query_llm(text: str) -> dict:
    prompt = (
        "You are a routing agent. Given the following user query:\n\n"
        f"\"{text}\"\n\n"
        "Decide if this query is directly related to medical symptoms where the user wants a disease prediction or if it is a general question. "
        "Respond only with one word: either 'disease_prediction' or 'general_query'."
    )
    try:
        result = llm_for_qa._generate([prompt])
        decision = result.generations[0][0].message.content.strip().lower()
    except Exception as e:
        decision = "general_query"
    if decision not in ["disease_prediction", "general_query"]:
        decision = "general_query"
    return {"branch": decision, "query": text}

# -----------------------
# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    # Use the LLM-based router to decide the branch.
    route_info = route_query_llm(user_input)
    
    if route_info["branch"] == "disease_prediction":
        # First, clean text & classify symptoms.
        cleaned_text = clean_text(route_info["query"])
        try:
            raw_symptoms = classify_symptoms_api(cleaned_text)
            print("Raw symptom response:", raw_symptoms)
            detected_symptoms = []
            # Handle various response formats from the symptom classifier.
            if isinstance(raw_symptoms, dict) and "label" in raw_symptoms and "confidences" in raw_symptoms:
                try:
                    confidence_val = raw_symptoms["confidences"][0].get("confidence", None)
                except (IndexError, KeyError):
                    confidence_val = None
                detected_symptoms = [(raw_symptoms["label"], confidence_val)]
            elif isinstance(raw_symptoms, dict):
                detected_symptoms = [(sym, conf) for sym, conf in raw_symptoms.items() if isinstance(conf, (float, int))]
            elif isinstance(raw_symptoms, list):
                if raw_symptoms and isinstance(raw_symptoms[0], dict) and "label" in raw_symptoms[0]:
                    detected_symptoms = [(item["label"], item.get("confidence", None)) for item in raw_symptoms]
                else:
                    detected_symptoms = [(s, None) for s in raw_symptoms]
            print("Detected symptoms:", detected_symptoms)
        except Exception as ex:
            return jsonify(response=f"Error in symptom classification: {ex}")
        
        preds = []
        if detected_symptoms:
            try:
                # Pass only the symptom labels to the disease predictor.
                symptom_labels = [s[0] for s in detected_symptoms]
                pred_result = predict_disease_api(symptom_labels)
                # Extract candidate diseases from the prediction result.
                candidate_diseases = []
                if isinstance(pred_result, dict):
                    if "confidences" in pred_result and isinstance(pred_result["confidences"], list):
                        candidate_diseases = [
                            candidate["label"] for candidate in pred_result["confidences"]
                            if isinstance(candidate, dict) and "label" in candidate
                        ]
                    elif "label" in pred_result and isinstance(pred_result["label"], str):
                        candidate_diseases = [pred_result["label"]]
                    else:
                        candidate_diseases = [
                            key for key in pred_result.keys() if key not in ["label", "confidences"]
                        ]
                preds = [(d, None) for d in candidate_diseases]
            except Exception as ex:
                return jsonify(response=f"Error in disease prediction: {ex}")
        
        if preds:
            # Build HTML list for diseases.
            diseases_list = "<ul>" + "".join(
                f"<li><strong>{d.upper()}</strong></li>" for d, _ in preds
            ) + "</ul>"
            descriptions = []
            for disease, _ in preds:
                info = retrieve_info_from_mongo(disease)
                if info["disease"]:
                    descriptions.append(markdown.markdown(info["description"].strip()) + "<br><br><br><br>")
            guidance_html = "".join(descriptions)
            collapsible_html = f"""
                <button class="toggle-btn" onclick="toggleDetails()">Details For Each Disease</button>
                <div id="disease-details" style="display: none; margin-top: 10px;">
                    {guidance_html}
                </div>
            """
            response_text = (
                f"Based on your symptoms, possible conditions are:<br>"
                f"{diseases_list}<br>{collapsible_html}"
            )
            return jsonify(response=response_text)
        else:
            # If no reliable symptom detection, fallback to general QA.
            qa_result = qa_chain.invoke({"query": cleaned_text})
            answer = qa_result.get("result", "I'm sorry, I couldn't find an answer to your question.")
            return jsonify(response=markdown.markdown(answer))
    
    elif route_info["branch"] == "general_query":
        qa_result = qa_chain.invoke({"query": route_info["query"]})
        answer = qa_result.get("result", "I'm sorry, I couldn't find an answer to your question.")
        return jsonify(response=markdown.markdown(answer))


if __name__ == "__main__":
    app.run(debug=True)
