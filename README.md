# 📧 Spam Email Classifier

AI-powered spam email detection using Machine Learning and Natural Language Processing.

## Features

- Dual ML models (Naive Bayes & Logistic Regression)
- Real-time classification
- TF-IDF vectorization
- Interactive web interface
- 95-98% accuracy

## Installation

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`

## Usage
```bash
streamlit run app.py
```

Access at: http://localhost:8501

## Technology Stack

- Python 3.8+
- Streamlit
- Scikit-learn
- NLTK
- Pandas & NumPy

## Project Structure
```
spam-email-classifier/
├── app.py
├── requirements.txt
├── utils/
│   └── preprocessing.py
├── models/
└── data/
```

## Author

[Vamshi]
```

---

### **STEP 8: Create .gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data
*.csv
*.pkl

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store

Thumbs.db
