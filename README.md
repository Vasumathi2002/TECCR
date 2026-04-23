🔍 Overview

TECCR (Tamil Emotion Contextual Cultural Reasoning) is a deep learning-based system designed to detect implicit cultural emotions in Tamil text. Unlike traditional sentiment analysis, TECCR identifies primary emotion, secondary emotion, and cultural context embedded in user-generated content.

The system integrates:

Deep Neural Networks (BiLSTM-based architecture)
Cultural reasoning layer
Web-based interaction platform (Admin + User modules)
🎯 Key Features
👤 User Module
User registration & authentication
View admin posts
Comment in Tamil
Emotion prediction performed silently
Ethical design: users cannot view predictions
👑 Admin Module
Post creation (text/image)
View and manage comments
Access emotion insights per comment
Filter by emotions (Sadness, Anger, Frustration, Sarcasm)
Trigger model training
🧠 TECCR Emotion Engine
Multi-output prediction:
Primary Emotion
Secondary Emotion
Cultural Context
Confidence-based interpretation
Batch and real-time prediction support
🏗️ System Architecture
Input (Tamil Text)
        ↓
Text Preprocessing & Tokenization
        ↓
BiLSTM-based Deep Learning Model
        ↓
Primary + Secondary Emotion Outputs
        ↓
TECCR Cultural Reasoning Layer
        ↓
Final Output (Emotion + Context + Confidence)
🧪 Model Details
Architecture: BiLSTM + Dense Multi-output Model
Embedding Layer + Dual LSTM layers
Outputs:
primary_output (Softmax)
secondary_output (Softmax)
Loss: Categorical Crossentropy
Optimizer: Adam
📊 Dataset
Tamil textual dataset with:
Text
Primary Emotion
Secondary Emotion
Cultural context annotations

Example:

Text: "மனசுக்குள்ள கேள்விகள் நிறைய"
Primary: Sadness
Secondary: Frustration
Context: Unresolved inner questions
🗄️ Database Schema

The system uses MySQL with the following tables:

users
admins
posts
comments
emotion_results
model_status

Includes:

Foreign key constraints
UTF-8 encoding for Tamil support
⚙️ Project Structure
├── app.py                  # Flask web application
├── train_model.py          # Model training pipeline
├── predict_emotion.py      # Emotion prediction engine
├── create_db.py            # Database setup script
├── emotion_count.py        # Dataset analysis script
├── dataset.csv             # Training dataset
├── models/                 # Saved model artifacts
│   ├── teccr_model.h5
│   ├── tokenizer.pkl
│   ├── encoders
│   └── metadata
🚀 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-username/TECCR.git
cd TECCR
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Setup Database
python create_db.py
4️⃣ Train Model
python train_model.py
5️⃣ Run Application
python app.py
🔮 Emotion Prediction Flow
User submits Tamil comment
Comment stored in database
Model predicts emotion
Results stored in emotion_results
Admin views analytics dashboard

Prediction includes:

Primary Emotion
Secondary Emotion
Cultural Context Explanation
Confidence Score
📈 Additional Tools
Emotion Distribution Analysis
python emotion_count.py

Provides:

Emotion frequency counts
Dataset summary statistics
⚖️ Ethical Considerations
Emotion predictions are hidden from users
Focus on cultural sensitivity & preservation
No misuse for profiling or bias amplification
Designed for research and assistive analytics
🔬 Research Contributions
Modeling implicit emotions in Tamil language
Cultural-aware NLP framework
Multi-label emotion prediction
Integration of deep learning + cultural reasoning
🌍 Tech Stack
Layer	Technology
Frontend	HTML, CSS, Bootstrap
Backend	Flask (Python)
ML/NLP	TensorFlow, BiLSTM
Database	MySQL
Language	Tamil NLP
📌 Future Scope
Hate speech detection
Multilingual expansion
Transformer-based models (IndicBERT, MuRIL)
Real-time analytics dashboard
