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
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, ChatGeneration, BaseRetriever, Document
from pydantic import Field

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_very_secret_key")

# MongoDB Connection (consider pooling/reusing connection if under high load)
mongodb_password = os.getenv("MONGODB_PASSWORD")
client_mongo = MongoClient(
    f"mongodb+srv://samratray:{mongodb_password}@cluster0.fcgztso.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
    serverSelectionTimeoutMS=5000  # 5 seconds timeout for Mongo connection
)
db = client_mongo["medichat"]
collection = db["disease_info"]

def retrieve_info_from_mongo(disease_name):
    try:
        doc = collection.find_one({"Disease": {"$regex": f"^{disease_name}$", "$options": "i"}})
        if doc:
            return {"disease": disease_name, "description": doc["Description"]}
    except Exception as e:
        print(f"MongoDB error: {e}")
    return {"disease": None, "description": "Information not found for the given disease."}

# Precompute the translation table once.
_translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
def clean_text(text: str) -> str:
    return " ".join(text.translate(_translator).split())

# Convert Markdown text to plain text.
def markdown_to_plain_text(md_text: str) -> str:
    html = markdown.markdown(md_text.strip())
    return BeautifulSoup(html, "html.parser").get_text()

# --- Retry mechanism for Gradio client initialization (for symptom and disease prediction) ---
def get_gradio_client(client_id: str, max_retries: int = 3, delay: int = 5):
    for attempt in range(max_retries):
        try:
            client = Client(client_id)
            if "faiss" not in client_id.lower():
                print(f"Loaded Gradio client '{client_id}' on attempt {attempt + 1} ✔")
            return client
        except httpx.ConnectTimeout as e:
            print(f"Attempt {attempt + 1} for client '{client_id}' timed out: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            print(f"Attempt {attempt + 1} for client '{client_id}' failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise RuntimeError(f"Could not load Gradio client {client_id} after {max_retries} attempts.")

# Initialize Gradio clients for symptom and disease prediction.
symptom_client = get_gradio_client("samratray/my_bert_space1")
disease_client = get_gradio_client("samratray/my_bert_space2")

# Initialize Gradio client for FAISS search.
faiss_client = Client("samratray/faiss")  # No logging for FAISS client initialization.

def classify_symptoms_api(text: str, threshold: float = 0.8):
    try:
        return symptom_client.predict(text=text, threshold=threshold, api_name="/predict")
    except Exception as ex:
        print(f"Error in classify_symptoms_api: {ex}")
        raise

def predict_disease_api(detected: list, threshold: float = 0.995):
    detected_string = ", ".join(detected)
    try:
        return disease_client.predict(
            detected=detected_string,
            threshold=threshold,
            output_format="dict",
            api_name="/predict"
        )
    except Exception as ex:
        print(f"Error in predict_disease_api: {ex}")
        raise

# Define a custom LLM that uses your Gemma2 API with exponential backoff.
class Gemma2LLM(BaseLLM):
    req_session: requests.Session = Field(default_factory=lambda: requests.Session())
    
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
        max_retries = 15  # Maximum retry attempts
        
        for prompt in prompts:
            payload = {"model": "gemma2-9b-it", "messages": [{"role": "user", "content": prompt}]}
            attempt = 0
            while attempt < max_retries:
                try:
                    response = self.req_session.post(url, headers=headers, json=payload, timeout=10)
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                    generations.append(ChatGeneration(message=AIMessage(content=content), generation_info={}))
                    break  # Success
                except requests.HTTPError as e:
                    if response.status_code == 429:
                        wait_time = 2 ** attempt  # Exponential backoff: 1,2,4,8...
                        print(f"Rate limit reached. Attempt {attempt + 1}/{max_retries}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        attempt += 1
                    else:
                        print(f"Error in Gemma2LLM _generate: {e}")
                        generations.append(ChatGeneration(message=AIMessage(content="Error generating response."), generation_info={}))
                        break
                except Exception as e:
                    print(f"Error in Gemma2LLM _generate: {e}")
                    generations.append(ChatGeneration(message=AIMessage(content="Error generating response."), generation_info={}))
                    break
            else:
                print("Max retries exhausted for prompt due to rate limiting.")
                generations.append(ChatGeneration(message=AIMessage(content="Error: Rate limit exceeded. Please try again later."), generation_info={}))
        return LLMResult(generations=[[g] for g in generations])
    
    class Config:
        arbitrary_types_allowed = True

# Instantiate LLMs.
llm_explanation = Gemma2LLM()
llm_for_qa = Gemma2LLM()

# ---------------------------
# Define a Dummy Retriever to satisfy the RetrievalQA chain.
# ---------------------------
class DummyRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str):
        return []
    @property
    def search_kwargs(self) -> dict:
        return {}

# ---------------------------
# Build a RetrievalQA Chain (for general queries)
# ---------------------------
prompt_template = (
    "You are MediChat, a specialized Medical Chatbot.\n "
    "Use the following context to answer the question in markdown. "
    "If the question is medically related, please provide a complete and thorough answer within your expertise. "
    "If the question falls outside the realm of medicine, respond with: "
    "'I am a Medical Chatbot and am not specialized in other domains. Please stick with my specialized domain.'\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
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
    "You are a routing agent. Your task is to analyze user queries and determine if the query is a request "
    "for a disease prediction based on provided symptoms or just a general query. "
    "Route the request to the 'disease_prediction' path only if the user appears to be asking for a diagnostic "
    "opinion or prediction based on the symptoms they provide.\n"
    "In all other cases, route it as a 'general_query'.\n"
    f"Query: \"{text}\"\n"
    "Respond with a single word: either 'disease_prediction' or 'general_query'."
    )

    try:
        result = llm_for_qa._generate([prompt])
        decision = result.generations[0][0].message.content.strip().lower()
        if decision not in ["disease_prediction", "general_query"]:
            print(f"Unexpected decision received: '{decision}'. Defaulting to 'general_query'.")
            decision = "general_query"
    except Exception as e:
        print(f"Error in route_query_llm: {e}. Defaulting to 'general_query'.")
        decision = "general_query"
    return {"branch": decision, "query": text}

# Updated decision function: now sending explanation, user query, and disease
def compute_decision_from_explanation(explanation: str, user_query: str, disease: str) -> bool:
    prompt_decision = (
        "Based on the following details, decide if the user's symptoms sufficiently match the given disease.\n\n"
        f"User Query: {user_query}\n"
        f"Disease: {disease}\n"
        f"Explanation: {explanation}\n\n"
        "Answer with a single word: 'Likely' if they match or 'Less Likely' if they do not."
    )
    try:
        decision_result = llm_explanation._generate([prompt_decision])
        output = decision_result.generations[0][0].message.content.strip().lower()
    except Exception as e:
        print(f"Error in compute_decision_from_explanation: {e}. Defaulting to 'Less Likely'.")
        return False
    if output.startswith("likely"):
        return True
    elif output.startswith("less likely"):
        return False
    else:
        print(f"Unexpected output in compute_decision_from_explanation: '{output}'. Defaulting to 'Less Likely'.")
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
    except Exception as e:
        print(f"LLM call for recheck failed: {e}")
        explanation_text = "Error generating explanation."
    decision = compute_decision_from_explanation(explanation_text, user_query, disease)
    return explanation_text, decision

# ---------------------------
# Helper: Call FAISS Space API using the Gradio client
# ---------------------------
def call_faiss_api(query: str) -> str:
    try:
        result = faiss_client.predict(query=query, api_name="/faiss_search")
        return result
    except Exception as e:
        error_msg = f"Error calling FAISS API: {e}"
        print(error_msg)
        return error_msg

# ---------------------------
# Flask Routes
# ---------------------------
@app.route("/")
def index():
    session.clear()
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    cleaned_text = clean_text(user_input)
    
    # Clear chat history if more than 5 minutes (300 seconds) have passed.
    now = time.time()
    last_time = session.get("chat_history_timestamp", now)
    if now - last_time > 300:
        session["chat_history"] = ""
        session["chat_history_timestamp"] = now
        print("Chat history cleared due to timeout.")
    
    chat_history = session.get("chat_history", "")
    chat_history += f"User: {user_input}\n"
    
    route_info = route_query_llm(user_input)
    structured_query = f"Conversation History:\n{chat_history}\nUser Query: {user_input}"
    response_text = ""
    
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
            sanitized_history = ""
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
                sanitized_history += f"{disease.upper()} ({possible_str})\n"
            response_text = f"Based on your symptoms, possible conditions are:<br><br>{diseases_html}"
            chat_history += f"Bot: {sanitized_history}\n"
        else:
            faiss_context = call_faiss_api(user_input)
            structured_query += f"\n\nContext from FAISS:\n{faiss_context}"
            qa_result = qa_chain.invoke({"query": structured_query})
            answer = qa_result.get("result", "I'm sorry, I couldn't find an answer.")
            response_text = markdown.markdown(answer)
            chat_history += f"Bot: {markdown_to_plain_text(response_text)}\n"
    else:
        faiss_context = call_faiss_api(user_input)
        structured_query += f"\n\nContext from FAISS:\n{faiss_context}"
        qa_result = qa_chain.invoke({"query": structured_query})
        answer = qa_result.get("result", "I'm sorry, I couldn't find an answer.")
        response_text = markdown.markdown(answer)
        chat_history += f"Bot: {markdown_to_plain_text(response_text)}\n"
    
    session["chat_history"] = chat_history
    session["chat_history_timestamp"] = time.time()
    
    return jsonify(response=response_text)

@app.route("/reset", methods=["GET"])
def reset():
    session.pop("chat_history", None)
    return redirect(url_for("index"))

# For production, gunicorn will load the `app` variable from this module.
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
