# MediChat - Medical Chatbot

A Flask-based medical chatbot developed by **utkarshroy** that leverages BERT models and LangChain to provide symptom classification, disease prediction, and general medical queries in real time.

---

## 🚀 Live Demo

> **https://medichat-deployment.onrender.com/**

---

## 📂 Repository Structure

├── app.py # Main Flask application
├── templates/
│ └── index.html # Front-end template
├── static/ # Static assets (CSS, JS)
├── requirements.txt # Python dependencies
├── .env # Environment variables (not committed)
└── README.md # This file

---

## 🔧 Features

- **Symptom Classification**: Uses BERT model (`utkarshroy/my_bert_space1`) via Gradio to detect symptoms from user input.
- **Disease Prediction**: Uses BERT model (`utkarshroy/my_bert_space2`) via Gradio to predict possible diseases from detected symptoms.
- **Gemma2 LLM**: Custom `Gemma2LLM` class for explanations and fallbacks using the Groq API.
- **RetrievalQA**: LangChain QA chain with a dummy retriever and optional FAISS search for general medical queries.
- **MongoDB Backend**: Stores disease descriptions in MongoDB Atlas, retrieved by name (case-insensitive).
- **Session Management**: Clears chat history after 5 minutes of inactivity.


---

## 🔧 Features

- **Symptom Classification**: Uses BERT model (`utkarshroy/my_bert_space1`) via Gradio to detect symptoms from user input.
- **Disease Prediction**: Uses BERT model (`utkarshroy/my_bert_space2`) via Gradio to predict possible diseases from detected symptoms.
- **Gemma2 LLM**: Custom `Gemma2LLM` class for explanations and fallbacks using the Groq API.
- **RetrievalQA**: LangChain QA chain with a dummy retriever and optional FAISS search for general medical queries.
- **MongoDB Backend**: Stores disease descriptions in MongoDB Atlas, retrieved by name (case-insensitive).
- **Session Management**: Clears chat history after 5 minutes of inactivity.

---

## 🔨 Installation

1. **Clone the repository**

```bash
git clone https://github.com/Utkarsh24-max/MediChat.git
cd MediChat

2. Create a virtual environment & install dependencies

bash
Copy code
python -m venv venv
venv\Scripts\activate     # Windows
# or source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
3. Set up environment variables

Copy .env.example to .env and add:

env
Copy code
FLASK_SECRET_KEY=your_flask_secret_key
MONGODB_PASSWORD=your_mongodb_password
GROQ_API_KEY=your_groq_api_key
4. Run the application

bash
Copy code
flask run
📋 Usage
Open http://localhost:5000 in your browser.

Type your symptoms or medical question in the chat box.

The bot will classify symptoms, predict diseases (with confidence levels), or answer general queries.

Click Reset or visit /reset to clear the session history.

🔌 API Endpoints
POST /chat : Send {"message": "<text>"} and receive { "response": "<HTML reply>" }.

GET /reset : Clears chat session and redirects to home.

⚙️ Configuration & Customization
Thresholds: Adjust threshold values for symptom classification (0.8) and disease prediction (0.995) in app.py.

Model IDs: Update Gradio client IDs if you rename or retrain your Hugging Face Spaces.

MongoDB: Modify database and collection names as needed in .env.

🗄️ Trained Models & Data
Download trained models, preprocessing scripts, and notebooks here:

https://drive.google.com/drive/folders/1RlgSh6kPphkHdVGlAaL9rdI86wcxuQ_I?usp=sharing

Link for the Data is given below:

https://drive.google.com/drive/folders/1pjnyJg79HXMuJid4fn0O_rwIe2vW38ZW?usp=sharing

🤝 Contributing
Contributions welcome—open issues or submit pull requests.

📜 License
Released under the MIT License.

Last updated: October 22, 2025
Developed by utkarshroy

