"""
QLingo - Authentication Module
Handles user login, signup, and session management
"""

import streamlit as st
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Path to user data file
USER_DATA_FILE = Path("data/users.json")

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if not USER_DATA_FILE.exists():
        USER_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Create with demo user
        demo_users = {
            "demo@qlingo.com": {
                "password": hash_password("demo123"),
                "name": "Demo User",
                "created_at": datetime.now().isoformat(),
                "role": "user"
            }
        }
        save_users(demo_users)
        return demo_users
    
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

def save_users(users_dict):
    """Save users to JSON file"""
    USER_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users_dict, f, indent=2)

def authenticate_user(email: str, password: str) -> bool:
    """Authenticate user credentials"""
    users = load_users()
    
    if email in users:
        stored_hash = users[email]['password']
        input_hash = hash_password(password)
        
        if stored_hash == input_hash:
            st.session_state.authenticated = True
            st.session_state.username = users[email]['name']
            return True
    
    return False

def register_user(email: str, password: str, name: str) -> tuple:
    """Register new user"""
    users = load_users()
    
    # Check if user exists
    if email in users:
        return False, "Email already registered!"
    
    # Add new user
    users[email] = {
        "password": hash_password(password),
        "name": name,
        "created_at": datetime.now().isoformat(),
        "role": "user"
    }
    
    save_users(users)
    return True, "Account created successfully! Please login."

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def show_login_page():
    """Display login/signup page"""
    
    # Custom CSS for login page
    st.markdown("""
    <style>
        .login-container {
            max-width: 450px;
            margin: 2rem auto;
            padding: 2rem;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .login-header h1 {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        
        .login-header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        .demo-info {
            background: #e7f3ff;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
            margin-top: 1rem;
            font-size: 0.9rem;
        }
        
        .feature-list {
            margin-top: 2rem;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .feature-item {
            padding: 0.5rem 0;
            color: #444;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Center content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-header">
            <h1>🌐 QLingo</h1>
            <p>AI-Powered Translation QA Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for login and signup
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            show_login_form()
        
        with tab2:
            show_signup_form()
        
        # Demo credentials info
        st.markdown("""
        <div class="demo-info">
            <strong>🎯 Demo Credentials:</strong><br>
            Email: demo@qlingo.com<br>
            Password: demo123
        </div>
        """, unsafe_allow_html=True)
        
        # Features section
        st.markdown("""
        <div class="feature-list">
            <strong>✨ Key Features:</strong>
            <div class="feature-item">✓ AI-powered semantic analysis</div>
            <div class="feature-item">✓ Multiple file format support</div>
            <div class="feature-item">✓ Comprehensive quality checks</div>
            <div class="feature-item">✓ Glossary management</div>
            <div class="feature-item">✓ Professional QA reports</div>
        </div>
        """, unsafe_allow_html=True)

def show_login_form():
    """Display login form"""
    with st.form("login_form"):
        email = st.text_input(
            "Email",
            placeholder="your.email@company.com",
            key="login_email"
        )
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            submit = st.form_submit_button("🔐 Login", use_container_width=True)
        
        if submit:
            if not email or not password:
                st.error("⚠️ Please fill in all fields!")
            else:
                with st.spinner("Authenticating..."):
                    if authenticate_user(email, password):
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password!")

def show_signup_form():
    """Display signup form"""
    with st.form("signup_form"):
        name = st.text_input(
            "Full Name",
            placeholder="John Doe",
            key="signup_name"
        )
        
        email = st.text_input(
            "Email",
            placeholder="your.email@company.com",
            key="signup_email"
        )
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Choose a strong password",
            key="signup_password"
        )
        
        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Re-enter your password",
            key="signup_confirm"
        )
        
        submit = st.form_submit_button("📝 Create Account", use_container_width=True)
        
        if submit:
            if not all([name, email, password, confirm_password]):
                st.error("⚠️ Please fill in all fields!")
            elif password != confirm_password:
                st.error("⚠️ Passwords don't match!")
            elif len(password) < 6:
                st.error("⚠️ Password must be at least 6 characters long!")
            elif '@' not in email:
                st.error("⚠️ Please enter a valid email address!")
            else:
                success, message = register_user(email, password, name)
                if success:
                    st.success(message)
                else:
                    st.error(message)