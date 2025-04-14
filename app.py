import os
import string
import time
import markdown
import requests
import httpx
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from pymongo import MongoClient
from dotenv import load_dotenv
from gradio_client import Client
from bs4 import BeautifulSoup  # Used to clean HTML

from langchain_core.language_models import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, ChatGeneration
from pydantic import Field

# -----------------------
load_dotenv()

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

# Function to convert Markdown text to clean plain text.
def markdown_to_plain_text(md_text: str) -> str:
    html = markdown.markdown(md_text.strip())
    return BeautifulSoup(html, "html.parser").get_text()

# --- Retry mechanism for Gradio client initialization ---
def get_gradio_client(client_id: str, max_retries: int = 3, delay: int = 5):
    for attempt in range(max_retries):
        try:
            client = Client(client_id)
            print(f"Loaded as API: {client_id} ✔")
            return client
        except httpx.ConnectTimeout as e:
            print(f"Attempt {attempt + 1} for {client_id} timed out: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise RuntimeError(f"Could not load {client_id} after {max_retries} attempts.")

# Initialize Gradio clients for symptom and disease prediction with retry.
symptom_client = get_gradio_client("samratray/my_bert_space1")
disease_client = get_gradio_client("samratray/my_bert_space2")

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
    
    def _generate(self, prompts: list[str], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None) -> LLMResult:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = "https://api.groq.com/openai/v1/chat/completions"
        generations = []
        for prompt in prompts:
            payload = {"model": "gemma2-9b-it", "messages": [{"role": "user", "content": prompt}]}
            response = self.req_session.post(url, headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            generations.append(ChatGeneration(message=AIMessage(content=content), generation_info={}))
        return LLMResult(generations=[[g] for g in generations])
    
    class Config:
        arbitrary_types_allowed = True

# Instantiate LLMs once.
llm_explanation = Gemma2LLM()
llm_for_qa = Gemma2LLM()

# Setup embeddings and FAISS index (load prebuilt index).
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_dir = "faiss_index"
if os.path.exists(index_dir) and os.listdir(index_dir):
    vector_store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
else:
    raise RuntimeError("FAISS index not found; build it first")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Original prompt template from your working code.
prompt_template = (
    "Use the following context to answer the question in markdown:\n"
    "(Provide a complete answer only if the question is medically related. If the question is outside the medical field, respond with 'I am a Medical Chatbot and am not specialized in other domains. Please stick with my specialized domain.' Always strive to be as helpful as possible within your area of expertise.)\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)
QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Build a RetrievalQA chain.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_for_qa,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

def route_query_llm(text: str) -> dict:
    # Use a simple heuristic if the query mentions key symptoms.
    lower_text = text.lower()
    if "chest pain" in lower_text or "heart attack" in lower_text or "pain" in lower_text:
        return {"branch": "disease_prediction", "query": text}
    
    # Otherwise, use the routing prompt.
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
    """
    Determines whether the disease is likely based on the generated explanation.
    This function makes an LLM call with a custom prompt asking whether the given explanation 
    supports the disease prediction, expecting a response of either 'Likely' or 'Less Likely'.

    Returns:
        bool: True if the decision is 'Likely', False otherwise.
    """
    prompt_decision = (
        "Based on the following explanation of how the symptoms match the disease:\n\n"
        f"Explanation: {explanation}\n\n"
        "Answer the following question with a single word: Do these symptoms support the disease prediction even a bit? "
        "Reply only 'Likely' if they do or 'Less Likely' if they do not, with no extra details."
    )
    
    try:
        # Call the LLM with the new prompt for decision making.
        decision_result = llm_explanation._generate([prompt_decision])
        output = decision_result.generations[0][0].message.content.strip().lower()
    except Exception as e:
        # In case of an exception, default to a conservative decision.
        print(f"LLM call for decision failed: {e}")
        return False

    # Interpret the LLM output.
    if output.startswith("likely"):
        return True
    elif output.startswith("less likely"):
        return False
    else:
        # Fallback: if the response is unclear, default to False.
        return False

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

@app.route("/")
def index():
    # Clear session data so a new session is started on each refresh.
    session.clear()
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    cleaned_text = clean_text(user_input)

    # Retrieve conversation history from session.
    chat_history = session.get("chat_history", "")
    chat_history += f"User: {user_input}\n"

    # Determine query type.
    route_info = route_query_llm(user_input)
    structured_query = f"Conversation History:\n{chat_history}\nUser Query: {user_input}"

    if route_info["branch"] == "disease_prediction":
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
                            <p><strong>Recheck Explanation:</strong> {clean_recheck_text}</p>
                        </div>
                    </div>
                """
            response_text = f"Based on your symptoms, possible conditions are:<br><br>{diseases_html}"
        else:
            qa_result = qa_chain.invoke({"query": structured_query})
            answer = qa_result.get("result", "I'm sorry, I couldn't find an answer.")
            response_text = markdown.markdown(answer)
    else:
        qa_result = qa_chain.invoke({"query": structured_query})
        answer = qa_result.get("result", "I'm sorry, I couldn't find an answer.")
        response_text = markdown.markdown(answer)

    chat_history += f"Bot: {response_text}\n"
    session["chat_history"] = chat_history

    return jsonify(response=response_text)

@app.route("/reset", methods=["GET"])
def reset():
    session.pop("chat_history", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
