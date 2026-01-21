"""
Spam Email Classifier - Complete ML Application
Real-time spam detection using Streamlit and Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import preprocessing utilities
from utils.preprocessing import download_nltk_data, preprocess_text

# Download NLTK data
download_nltk_data()

# Create necessary directories
Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# ============================================================================
# DATA LOADING FUNCTION
# ============================================================================

@st.cache_data
def load_sample_data():
    """Load or create sample spam/ham dataset"""
    
    spam_emails = [
        "Congratulations! You've won a $1000 gift card. Click here to claim now!",
        "URGENT: Your account will be closed. Verify your identity immediately.",
        "Make money fast! Work from home and earn $5000 per week.",
        "You have been selected for a free vacation to the Bahamas. Act now!",
        "Viagra and cialis at lowest prices. Buy now without prescription.",
        "Dear friend, I am a prince from Nigeria and need your help transferring funds.",
        "Click here for free iPhone! Limited offer, only today!",
        "Lose 20 pounds in 2 weeks with this amazing diet pill.",
        "Your PayPal account has been suspended. Click to reactivate.",
        "Hot singles in your area want to meet you tonight!",
        "Get a loan approved in 24 hours with bad credit. Apply now!",
        "You've been pre-approved for a credit card with $10,000 limit.",
        "Work from home stuffing envelopes. Make $3000 weekly guaranteed!",
        "Buy cheap software. Windows, Office, Adobe 90% off!",
        "Enlarge your manhood naturally. No pills needed.",
        "Free casino bonus! Claim your $500 welcome package now.",
        "Act now! This offer expires in 1 hour. Don't miss out!",
        "You have unclaimed inheritance. Contact us to receive $2 million.",
        "Get rich quick with cryptocurrency trading. 100% guaranteed returns.",
        "Your computer is infected! Download antivirus now.",
    ]
    
    ham_emails = [
        "Hi John, can we schedule a meeting tomorrow at 2pm to discuss the project?",
        "Thank you for your email. I will review the documents and get back to you.",
        "The quarterly report is attached. Please let me know if you have questions.",
        "Reminder: Team lunch tomorrow at noon in the cafeteria.",
        "Could you please send me the presentation slides from last week's meeting?",
        "I've completed the analysis you requested. The results are very interesting.",
        "Happy birthday! Hope you have a wonderful day with family and friends.",
        "The software update has been scheduled for this weekend.",
        "Please find attached the invoice for last month's services.",
        "Thanks for your help with the client presentation yesterday.",
        "I'm running a few minutes late. Will be there by 3:15pm.",
        "The conference call has been rescheduled to Friday at 10am.",
        "Great job on the report! Your analysis was very thorough.",
        "Could you review this document before I send it to the client?",
        "The server maintenance is complete. All systems are back online.",
        "Looking forward to working with you on the new project.",
        "Please confirm your attendance for the training session next week.",
        "I've updated the spreadsheet with the latest sales figures.",
        "Thanks for bringing that issue to my attention. I'll look into it.",
        "The package you ordered has been shipped and will arrive tomorrow.",
    ]
    
    # Create DataFrame
    emails = spam_emails + ham_emails
    labels = ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails)
    
    df = pd.DataFrame({'email': emails, 'label': labels})
    
    # Save to CSV
    df.to_csv('data/emails.csv', index=False)
    
    return df

# ============================================================================
# MODEL TRAINING FUNCTION
# ============================================================================

@st.cache_resource
def train_models():
    """Train and cache ML models"""
    
    # Load data
    df = load_sample_data()
    
    # Preprocess emails
    df['processed_email'] = df['email'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_email'], 
        df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Multinomial Naive Bayes
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_tfidf, y_train)
    nb_predictions = nb_model.predict(X_test_tfidf)
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_model.fit(X_train_tfidf, y_train)
    lr_predictions = lr_model.predict(X_test_tfidf)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    
    # Save models
    with open('models/nb_model.pkl', 'wb') as f:
        pickle.dump(nb_model, f)
    
    with open('models/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    models = {
        'vectorizer': vectorizer,
        'naive_bayes': nb_model,
        'logistic_regression': lr_model,
        'nb_accuracy': nb_accuracy,
        'lr_accuracy': lr_accuracy
    }
    
    return models

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_email(email_text, model_choice, models):
    """
    Predict if an email is spam or ham
    
    Args:
        email_text (str): Email content
        model_choice (str): Selected model name
        models (dict): Dictionary of trained models
        
    Returns:
        tuple: (prediction, confidence, probabilities)
    """
    # Preprocess
    processed_text = preprocess_text(email_text)
    
    # Vectorize
    text_tfidf = models['vectorizer'].transform([processed_text])
    
    # Select model
    if model_choice == "Naive Bayes":
        model = models['naive_bayes']
    else:
        model = models['logistic_regression']
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = max(probabilities) * 100
    
    return prediction, confidence, probabilities

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Page config
    st.set_page_config(
        page_title="Spam Email Classifier",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }
        .spam-box {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
            padding: 1.5rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .ham-box {
            background-color: #e8f5e9;
            border-left: 5px solid #4caf50;
            padding: 1.5rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">📧 Spam Email Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Real-Time Email Detection using Machine Learning</p>', unsafe_allow_html=True)
    
    # Train models (cached)
    with st.spinner('🔄 Training models... Please wait.'):
        models = train_models()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "Select ML Model:",
            ["Naive Bayes", "Logistic Regression"],
            help="Choose between speed (NB) or accuracy (LR)"
        )
        
        st.markdown("---")
        
        # Display model performance
        st.subheader("📊 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Naive Bayes", f"{models['nb_accuracy']*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Logistic Reg", f"{models['lr_accuracy']*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # About section
        st.subheader("ℹ️ About")
        st.info("""
        This application uses Machine Learning to classify emails as spam or legitimate.
        
        **Features:**
        - TF-IDF vectorization
        - Two ML models
        - Real-time predictions
        - Confidence scores
        """)
        
        st.markdown("---")
        
        # Sample emails
        st.subheader("💡 Quick Test")
        
        if st.button("📩 Legitimate Email", use_container_width=True):
            st.session_state.sample = "Hi Sarah, can we schedule a meeting tomorrow at 2pm to discuss the quarterly report? Please let me know if this time works for you. Thanks, John"
        
        if st.button("🚫 Spam Email", use_container_width=True):
            st.session_state.sample = "CONGRATULATIONS!!! You've WON $1,000,000! Click here NOW to claim your prize! This offer expires in 24 hours. Act fast!"
        
        if st.button("⚠️ Phishing Email", use_container_width=True):
            st.session_state.sample = "URGENT: Your account has been suspended due to suspicious activity. Verify your identity immediately by clicking here or your account will be permanently closed within 24 hours."
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✉️ Enter Email Content")
        
        # Get sample if selected
        default_text = st.session_state.get('sample', '')
        
        email_input = st.text_area(
            "Paste or type email text here:",
            value=default_text,
            height=200,
            placeholder="Enter email text for classification...",
            help="Enter the complete email content including subject and body"
        )
        
        # Character count
        if email_input:
            st.caption(f"📝 Character count: {len(email_input)}")
        
        # Buttons
        col_btn1, col_btn2 = st.columns([3, 1])
        
        with col_btn1:
            classify_btn = st.button("🔍 Classify Email", type="primary", use_container_width=True)
        
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.sample = ''
                st.rerun()
        
        # Classification logic
        if classify_btn:
            if email_input.strip():
                with st.spinner('🔄 Analyzing email...'):
                    prediction, confidence, probabilities = predict_email(
                        email_input, 
                        model_choice, 
                        models
                    )
                
                # Display result
                st.markdown("### 📋 Classification Result")
                
                if prediction == 'spam':
                    st.markdown(f"""
                        <div class="spam-box">
                            <h2 style="color: #f44336; margin: 0;">🚫 SPAM DETECTED</h2>
                            <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                                <strong>Confidence: {confidence:.2f}%</strong>
                            </p>
                            <p style="margin-top: 1rem; color: #666;">
                                Model: {model_choice}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="ham-box">
                            <h2 style="color: #4caf50; margin: 0;">✅ LEGITIMATE EMAIL</h2>
                            <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                                <strong>Confidence: {confidence:.2f}%</strong>
                            </p>
                            <p style="margin-top: 1rem; color: #666;">
                                Model: {model_choice}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.markdown("### 📊 Probability Distribution")
                
                prob_df = pd.DataFrame({
                    'Class': ['Legitimate (Ham)', 'Spam'],
                    'Probability': [probabilities[0]*100, probabilities[1]*100]
                })
                
                st.bar_chart(prob_df.set_index('Class'))
                
                # Detailed metrics
                st.markdown("### 📈 Detailed Analysis")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.metric("Ham Probability", f"{probabilities[0]*100:.2f}%")
                
                with col_m2:
                    st.metric("Spam Probability", f"{probabilities[1]*100:.2f}%")
                
                with col_m3:
                    st.metric("Word Count", len(email_input.split()))
                
            else:
                st.warning("⚠️ Please enter email text to classify.")
    
    with col2:
        st.subheader("📖 How It Works")
        
        st.markdown("""
        **Step 1: Input**
        Enter your email text
        
        **Step 2: Preprocessing**
        Text is cleaned and tokenized
        
        **Step 3: Feature Extraction**
        TF-IDF vectorization (500 features)
        
        **Step 4: Classification**
        ML model predicts spam/ham
        
        **Step 5: Result**
        View confidence and probabilities
        """)
        
        st.markdown("---")
        
        st.subheader("🎯 Model Info")
        st.info(f"""
        **Currently Using:** {model_choice}
        
        **Accuracy:** {models['nb_accuracy']*100:.2f}% (NB) | {models['lr_accuracy']*100:.2f}% (LR)
        
        **Features:** 500 TF-IDF dimensions
        
        **Training:** 80-20 split
        """)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()