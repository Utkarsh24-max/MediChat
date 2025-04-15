import os
import string
import time
import markdown
import requests
import httpx
import functools
import hashlib

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from pymongo import MongoClient
from dotenv import load_dotenv
from gradio_client import Client
from bs4 import BeautifulSoup  # Used to clean HTML

from langchain_core.language_models import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, ChatGeneration, BaseRetriever, Document
from pydantic import Field

load_dotenv()

# --- Simple Cache Decorator ---
cache = {}

def cache_llm(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key_raw = (args, kwargs)
        key = hashlib.sha256(repr(key_raw).encode('utf-8')).hexdigest()
        if key in cache:
            return cache[key]
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_very_secret_key")

# MongoDB Connection
mongodb_password = os.getenv("MONGODB_PASSWORD")
client_mongo = MongoClient(
    f"mongodb+srv://samratray:{mongodb_password}@cluster0.fcgztso.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
db = client_mongo["medichat"]
collection = db["disease_info"]

def retrieve_info_from_mongo(disease_name):
    doc = collection.find_one({"Disease": {"$regex": f"^{disease_name}$", "$options": "i"}})
    if doc:
        return {"disease": disease_name, "description": doc["Description"]}
    return {"disease": None, "description": "Information not found for the given disease."}

# Precompute the translation table once.
_translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
def clean_text(text: str) -> str:
    return " ".join(text.translate(_translator).split())

def markdown_to_plain_text(md_text: str) -> str:
    html = markdown.markdown(md_text.strip())
    return BeautifulSoup(html, "html.parser").get_text()

# --- Retry mechanism for Gradio client initialization (for symptom and disease prediction) ---
def get_gradio_client(client_id: str, max_retries: int = 3, delay: int = 5):
    for attempt in range(max_retries):
        try:
            # Optionally adjust timeout settings here if needed.
            client = Client(client_id)
            if "faiss" not in client_id.lower():
                print(f"Loaded as API: {client_id} ✔")
            return client
        except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            print(f"Attempt {attempt + 1} for {client_id} timed out: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise RuntimeError(f"Could not load {client_id} after {max_retries} attempts.")

# Initialize Gradio clients.
symptom_client = get_gradio_client("samratray/my_bert_space1")
disease_client = get_gradio_client("samratray/my_bert_space2")
faiss_client = Client("samratray/faiss")  # For FAISS search.

def classify_symptoms_api(text: str, threshold: float = 0.8):
    return symptom_client.predict(text=text, threshold=threshold, api_name="/predict")

def predict_disease_api(detected: list, threshold: float = 0.995):
    detected_string = ", ".join(detected)
    return disease_client.predict(
        detected=detected_string,
        threshold=threshold,
        output_format="dict",
        api_name="/predict"
    )

# Define a custom LLM that uses your Gemma2 API.
class Gemma2LLM(BaseLLM):
    req_session: requests.Session = Field(default_factory=requests.Session)
    
    @property
    def _llm_type(self) -> str:
        return "gemma2"
    
    @cache_llm
    def _generate(self, prompts: list[str], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None) -> LLMResult:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = "https://api.groq.com/openai/v1/chat/completions"
        generations = []
        for prompt in prompts:
            time.sleep(0.3)  # Minimal debouncing.
            for attempt in range(2):  # Try up to two times.
                try:
                    payload = {"model": "gemma2-9b-it", "messages": [{"role": "user", "content": prompt}]}
                    response = self.req_session.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                    generations.append(ChatGeneration(message=AIMessage(content=content), generation_info={}))
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        print(f"Rate limited. Attempt {attempt+1} for prompt: {prompt}")
                        time.sleep(1)
                    else:
                        raise
        return LLMResult(generations=[[g] for g in generations])
    
    class Config:
        arbitrary_types_allowed = True

llm_explanation = Gemma2LLM()
llm_for_qa = Gemma2LLM()

# Dummy retriever for the QA chain.
class DummyRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str):
        return []
    @property
    def search_kwargs(self) -> dict:
        return {}

prompt_template = (
    "Use the following context to answer the question in markdown:\n"
    "(Provide a complete answer only if the question is medically related. If the question is outside the medical field, respond with 'I am a Medical Chatbot and am not specialized in other domains. Please stick with my specialized domain.' Always strive to be as helpful as possible within your area of expertise.)\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)
QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_for_qa,
    chain_type="stuff",
    retriever=DummyRetriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

def route_query_llm(text: str) -> dict:
    prompt = (
        "You are a routing agent. Decide if the query is related to medical symptoms (for disease prediction) "
        "or if it is a general query.\n"
        f"Query: \"{text}\"\n"
        "Respond with either 'disease_prediction' or 'general_query'."
    )
    try:
        result = llm_for_qa._generate([prompt])
        decision = result.generations[0][0].message.content.strip().lower()
    except Exception:
        decision = "general_query"
    if decision not in ["disease_prediction", "general_query"]:
        decision = "general_query"
    return {"branch": decision, "query": text}

def compute_decision_from_explanation(explanation: str) -> bool:
    prompt_decision = (
        "Based on the following explanation of how the symptoms match the disease:\n\n"
        f"Explanation: {explanation}\n\n"
        "Answer with a single word: Reply only 'Likely' if they do or 'Less Likely' if they do not."
    )
    try:
        decision_result = llm_explanation._generate([prompt_decision])
        output = decision_result.generations[0][0].message.content.strip().lower()
    except Exception as e:
        print(f"LLM call for decision failed: {e}")
        return False
    return output.startswith("likely")

def recheck_and_decide_disease(symptoms: list, disease: str, user_query: str) -> (str, bool):
    prompt_explanation = (
        f"How likely is the disease based on the given symptoms:\n"
        f"User Query: {user_query}\n"
        f"Disease: {disease}\n"
        "Provide a brief explanation about why the disease does or does not match the symptoms."
    )
    try:
        explanation_result = llm_explanation._generate([prompt_explanation])
        explanation_text = explanation_result.generations[0][0].message.content.strip()
    except Exception:
        explanation_text = "Error generating explanation."
    decision = compute_decision_from_explanation(explanation_text)
    return explanation_text, decision

def call_faiss_api(query: str) -> str:
    try:
        result = faiss_client.predict(query=query, api_name="/faiss_search")
        return result
    except Exception as e:
        return f"Error calling FAISS API: {e}"

# To reduce cookie size, we define a maximum message length.
max_message_length = 300  # Adjust as needed.

def truncate_message(message: str) -> str:
    return message if len(message) <= max_message_length else message[:max_message_length] + "..."

@app.route("/")
def index():
    session["chat_history"] = []  # Initialize as an empty list.
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    cleaned_text = clean_text(user_input)
    
    # Load the conversation history as a list.
    chat_history = session.get("chat_history", [])
    chat_history.append(f"User: {truncate_message(user_input)}")
    
    # Use only the last five exchanges.
    recent_history = chat_history[-5:]
    structured_query = "Conversation History:\n" + "\n".join(recent_history) + f"\nUser Query: {truncate_message(user_input)}"
    
    if route_query_llm(user_input)["branch"] == "disease_prediction":
        try:
            raw_symptoms = classify_symptoms_api(cleaned_text)
        except Exception as ex:
            return jsonify(response=f"Error in symptom classification: {ex}")
        detected_symptoms = []
        if isinstance(raw_symptoms, dict):
            if "label" in raw_symptoms and "confidences" in raw_symptoms:
                detected_symptoms = [item["label"] for item in raw_symptoms["confidences"] if isinstance(item, dict)]
            else:
                detected_symptoms = [sym for sym, conf in raw_symptoms.items() if isinstance(conf, (float, int))]
        elif isinstance(raw_symptoms, list):
            if raw_symptoms and isinstance(raw_symptoms[0], dict) and "label" in raw_symptoms[0]:
                detected_symptoms = [item["label"] for item in raw_symptoms]
            else:
                detected_symptoms = raw_symptoms

        preds = []
        if detected_symptoms:
            try:
                pred_result = predict_disease_api(detected_symptoms)
            except Exception as ex:
                return jsonify(response=f"Error in disease prediction: {ex}")
            candidate_diseases = []
            if isinstance(pred_result, dict):
                if "confidences" in pred_result and isinstance(pred_result["confidences"], list):
                    candidate_diseases = [c["label"] for c in pred_result["confidences"]
                                          if isinstance(c, dict) and "label" in c]
                elif "label" in pred_result and isinstance(pred_result["label"], str):
                    candidate_diseases = [pred_result["label"]]
                else:
                    candidate_diseases = [key for key in pred_result.keys() if key not in ["label", "confidences"]]
            for disease in candidate_diseases:
                try:
                    explanation, decision = recheck_and_decide_disease(detected_symptoms, disease, user_input)
                    preds.append((disease, explanation, decision))
                    print(f"Processed Candidate: {disease} | Decision: {decision}")
                except Exception as ex:
                    print(f"Error processing {disease}: {ex}")
        preds.sort(key=lambda x: not x[2])
        if preds:
            diseases_html = ""
            for disease, recheck_text, decision in preds:
                info = retrieve_info_from_mongo(disease)
                description_html = markdown.markdown(info["description"].strip())
                clean_recheck_text = markdown_to_plain_text(recheck_text)
                possible_str = "Likely" if decision else "Less Likely"
                diseases_html += f"""
                    <div class="disease-entry" style="margin-bottom:20px;">
                        <strong>{disease.upper()}</strong>
                        <span style="margin-left:20px;">({possible_str})</span>
                        <button class="toggle-btn" onclick="toggleDetails('{disease}')">Details</button>
                        <div id="details-{disease}" class="disease-details" style="display:none; margin-top: 10px;">
                            <p>{description_html}</p>
                            <br/><br/>
                            <p><strong>Recheck Explanation:</strong> {truncate_message(markdown_to_plain_text(clean_recheck_text))}</p>
                        </div>
                    </div>
                """
            response_text = f"Based on your symptoms, possible conditions are:<br><br>{diseases_html}"
        else:
            faiss_context = call_faiss_api(user_input)
            structured_query += f"\n\nContext from FAISS:\n{faiss_context}"
            qa_result = qa_chain.invoke({"query": structured_query})
            answer = qa_result.get("result", "I'm sorry, I couldn't find an answer.")
            response_text = markdown.markdown(answer)
    else:
        faiss_context = call_faiss_api(user_input)
        structured_query += f"\n\nContext from FAISS:\n{faiss_context}"
        qa_result = qa_chain.invoke({"query": structured_query})
        answer = qa_result.get("result", "I'm sorry, I couldn't find an answer.")
        response_text = markdown.markdown(answer)
    
    # Prepare a clean version for logging and storing.
    clean_response = markdown_to_plain_text(response_text)
    chat_history.append(f"Bot: {truncate_message(clean_response)}")
    session["chat_history"] = chat_history[-5:]
    
    return jsonify(response=response_text)

@app.route("/reset", methods=["GET"])
def reset():
    session.pop("chat_history", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)