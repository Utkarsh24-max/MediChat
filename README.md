# MediChat - Medical Chatbot

A Flask-based medical chatbot that leverages BERT models and LangChain to provide symptom classification, disease prediction, and general medical queries in real time.

---

## 🚀 Live Demo

> **https://medichat-deployment.onrender.com/**

---

## 📂 Repository Structure

```
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # Front-end template
├── static/                # Static assets (CSS, JS)
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not committed)
└── README.md              # This file
```

---

## 🔧 Features

- **Symptom Classification**: Uses BERT model (`samratray/my_bert_space1`) via Gradio to detect symptoms from user input.
- **Disease Prediction**: Uses BERT model (`samratray/my_bert_space2`) via Gradio to predict possible diseases from detected symptoms.
- **Gemma2 LLM**: Custom `Gemma2LLM` class for explanations and fallbacks using the Groq API.
- **RetrievalQA**: LangChain QA chain with a dummy retriever and optional FAISS search for general medical queries.
- **MongoDB Backend**: Stores disease descriptions in MongoDB Atlas, retrieved by name (case-insensitive).
- **Session Management**: Clears chat history after 5 minutes of inactivity.

---

## 🔨 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/SamratRay2005/MediChat_deployment.git
   cd MediChat_deployment
   ```

2. **Create a virtual environment & install dependencies**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or venv\Scripts\activate  # Windows

   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Copy `.env.example` to `.env` and add:

   ```env
   FLASK_SECRET_KEY=your_flask_secret_key
   MONGODB_PASSWORD=your_mongodb_password
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run the application**

   ```bash
   flask run
   ```

---

## 📋 Usage

- Open `http://localhost:5000` in your browser.
- Type your symptoms or medical question in the chat box.
- The bot will classify symptoms, predict diseases (with confidence levels), or answer general queries.
- Click **Reset** or visit `/reset` to clear the session history.

---

## 🔌 API Endpoints

- `POST /chat` : Send `{"message": "<text>"}` and receive `{ "response": "<HTML reply>" }`.
- `GET /reset` : Clears chat session and redirects to home.

---

## ⚙️ Configuration & Customization

- **Thresholds**: Adjust `threshold` values for symptom classification (`0.8`) and disease prediction (`0.995`) in `app.py`.
- **Model IDs**: Update Gradio client IDs if you rename or retrain your Hugging Face Spaces.
- **MongoDB**: Modify database and collection names as needed in `.env`.

---

## 🗄️ Trained Models & Data

Download trained models, preprocessing scripts, and notebooks here:

[Google Drive Folder](https://drive.google.com/drive/folders/1RlgSh6kPphkHdVGlAaL9rdI86wcxuQ_I?usp=sharing)

---

## 🤝 Contributing

Contributions welcome—open issues or submit pull requests.

---

## 📜 License

Released under the [MIT License](LICENSE).

---

_Last updated: April 27, 2025_

