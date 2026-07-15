# app.py — FINAL STABLE WORKING VERSION (ACTUALLY FIXED)

import sys
print("PYTHON USED:", sys.executable)

from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import os
import re
import traceback
import hashlib
from datetime import datetime
import json
from difflib import SequenceMatcher

# MODEL imports
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except Exception:
    GPT2LMHeadModel = None
    GPT2Tokenizer = None

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# -------------------------
# SESSION MEMORY
# -------------------------
SESSION_MEMORY = {}

MAX_SESSION_TURNS = 8

# -------------------------
# Database
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -------------------------
# Models
# -------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    role = db.Column(db.String(10), nullable=False)
    text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def get_recent_user_messages(user_id, limit=2):
    """Get recent user messages for context"""
    try:
        msgs = (Conversation.query
                       .filter_by(user_id=user_id, role="user")
                       .order_by(Conversation.created_at.desc())
                       .limit(limit)
                       .all())
        # Return list of message texts, not Conversation objects
        return [msg.text for msg in msgs]
    except Exception:
        return []

def get_last_bot_reply(user_id):
    """Get last bot reply for fallback comparison"""
    try:
        last_bot = (Conversation.query
                     .filter_by(user_id=user_id, role="serene")
                     .order_by(Conversation.created_at.desc())
                     .first())
        return last_bot.text if last_bot else ""
    except Exception:
        return ""

def get_recent_context(user_id, limit=8):
    """Get recent conversation context"""
    try:
        context = (Conversation.query
                   .filter_by(user_id=user_id)
                   .order_by(Conversation.created_at.desc())
                   .limit(limit)
                   .all())
        return "\n".join([c.text for c in context])
    except Exception:
        return ""

def retrieve_relevant_knowledge(user_message, user_id, k=3):
    """Retrieve top-k most relevant knowledge lines with context awareness"""
    try:
        # Get last 2 user messages for context
        recent_user_msgs = get_recent_user_messages(user_id, limit=2)
        
        # recent_user_msgs is already a list of strings
        context = " ".join(recent_user_msgs).lower()
        
        # Score all knowledge lines
        scored_knowledge = []
        for line in KNOWLEDGE_BASE:
            score = calculate_retrieval_score(user_message, line)
            
            # Boost score if context matches
            if any(word in line.lower() for word in context.split()):
                score += 1
            
            scored_knowledge.append((score, line))
        
        # Sort by score and return top-k
        scored_knowledge.sort(reverse=True)
        return [line for score, line in scored_knowledge[:k]]
        
    except Exception:
        return []

def calculate_retrieval_score(query, knowledge_line):
    """Calculate weighted score for knowledge retrieval"""
    query_l = query.lower()
    knowledge_l = knowledge_line.lower()
    
    score = 0
    
    # Exact keyword match (+3)
    if any(word in knowledge_l for word in re.findall(r'\b\w+\b', query_l)):
        score += 3
    
    # Similar word match (+2)
    matcher = SequenceMatcher(None, query_l)
    matcher.set_seq2(knowledge_l)
    if matcher.ratio() > 0.6:
        score += 2
    
    # Partial match (+1)
    if any(word in knowledge_l for word in query_l.split()):
        score += 1
    
    return score

def get_relevant_knowledge(msg):
    """Simple knowledge retrieval for fallback"""
    try:
        # Simple keyword matching for fallback
        msg_l = msg.lower()
        for line in KNOWLEDGE_BASE:
            if any(word in line.lower() for word in msg_l.split() if len(word) > 2):
                return line
        return ""
    except Exception:
        return ""

# -------------------------
# RAG SYSTEM - Enhanced Retrieval
# -------------------------
KNOWLEDGE_BASE = [
    "When friends betray you, it cuts deep because trust is foundational to friendship. The pain feels different from other conflicts.",
    "Secrets between friends create distance because honesty is core to real connection.",
    "Family conflicts hurt more because these people shaped your early understanding of love and support.",
    "Anger is often masking deeper feelings like hurt, fear, or disappointment.",
    "Sadness that persists might be depression - your mind deserves rest and support.",
    "Anxiety often comes from feeling overwhelmed or uncertain about the future.",
    "Loneliness feels most painful when surrounded by people who don't understand.",
    "Trust, once broken, takes consistent effort and genuine change to rebuild.",
    "Grief comes in waves - some days are okay, others feel impossible.",
    "Stress about work often bleeds into personal life because it consumes so much time.",
    "Decision paralysis happens when you're overwhelmed by options and fear making wrong choice.",
    "Self-care isn't selfish - it's necessary for mental health and helping others."
]

def generate_model_response(user_message, user_id):
    """Generate response using GPT-2 model with RAG-enhanced context"""
    
    if not model or not tokenizer:
        return None

    try:
        # Get recent conversation context
        context = get_recent_context(user_id, 8)

        # Retrieve relevant knowledge
        relevant_knowledge = retrieve_relevant_knowledge(user_message, user_id, k=3)
        knowledge_context = "\n".join(relevant_knowledge) if relevant_knowledge else ""

        # Build enhanced prompt with RAG
        prompt = f"""You are Serene, a calm and empathetic mental health support companion.

Respond naturally to what the user says.
Acknowledge their feelings and experiences.
Use relevant insights to support, but don't sound robotic.
Keep the conversation flowing like a real supportive person.

Relevant emotional context:
{knowledge_context}

Conversation so far:
{context}

User: {user_message}
Serene:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        reply = text.split("Serene:")[-1].strip()
        reply = re.split(r"User:", reply)[0].strip()

        if len(reply) < 3:
            return None

        return reply

    except Exception as e:
        print("MODEL ERROR:", e)
        return None

# -------------------------
# SMART LOGIC (FIXED)
# -------------------------
def smart_override(msg, user_id):
    m = msg.lower()
    recent = get_recent_user_messages(user_id)
    # Convert list of messages to string
    recent_str = " ".join(recent) if recent else ""
    combined = m + " " + recent_str

    # DECISION (STRONG FIX)
    if any(q in m for q in ["should i", "what should i do", "what do you think"]):
        if "friend" in combined:
            return (
                "🌸 I can see why you're torn. They hurt you, but you're also scared of losing them.\n\n"
                "You don’t have to decide everything right now. You could take a little space "
                "or talk to them honestly and see how they respond.\n\n"
                "What matters is whether they’re willing to understand your feelings."
            )

    # BETRAYAL CONTEXT
    if "friend" in combined and ("betray" in combined or "secret" in combined):
        return (
            "🌸 That kind of betrayal really hurts—especially when it comes from people you trust."
        )

    # FEAR OF LOSING
    if "what if" in m and "friend" in combined:
        return (
            "🌸 That fear makes a lot of sense. Losing people you care about can feel really scary."
        )

    # ALONE
    if "no one" in combined or "alone" in combined:
        return (
            "🌸 That’s what makes this so much harder… it’s not just about them, it’s about feeling alone."
        )

    return None

# -------------------------
# FALLBACK (IMPROVED)
# -------------------------
def fallback_response(msg, user_id):
    m = msg.lower()
    last = get_last_bot_reply(user_id).lower()

    if "sad" in m:
        return "🌸 I’m really sorry you’re feeling this way. What’s been weighing on you the most?"

    if "upset" in m:
        if "hurting the most" in last:
            return "🌸 That sounds really painful… especially since so many people were involved."
        return "🌸 That sounds really upsetting. What about it is hurting the most?"

    return "🌸 I hear you. That sounds really difficult."

# -------------------------
# SAFETY
# -------------------------
SAFETY_KEYWORDS = ["suicide", "kill myself", "i want to die"]

SAFETY_RESPONSE = "🌸 I'm really sorry you're feeling this way. Please call 112."

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    if "user_id" in session:
        return render_template("index.html")
    return redirect(url_for("login"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        if User.query.filter_by(username=username).first():
            return "Username exists", 400

        user = User(username=username,
                    password=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            return redirect(url_for("home"))

        return "Invalid credentials", 401

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# -------------------------
# CHAT (FINAL FLOW FIX)
# -------------------------
@app.route("/get", methods=["POST"])
def get_bot_response():
    if "user_id" not in session:
        return "Login first", 401

    user_id = session["user_id"]
    msg = request.form.get("msg", "").strip()

    if not msg:
        return "Say something"

    db.session.add(Conversation(user_id=user_id, role="user", text=msg))
    db.session.commit()

    msg_lower = msg.lower()

    if any(k in msg_lower for k in SAFETY_KEYWORDS):
        reply = SAFETY_RESPONSE
    else:
        # 1️⃣ Smart reasoning FIRST
        reply = smart_override(msg, user_id)

        # 2️⃣ Fallback SECOND (IMPORTANT CHANGE)
        if not reply:
            reply = fallback_response(msg, user_id)

        # 3️⃣ RAG LAST (only if still empty)
        if not reply:
            knowledge = get_relevant_knowledge(msg)
            if knowledge:
                reply = f"🌸 {knowledge.splitlines()[0]}"

    db.session.add(Conversation(user_id=user_id, role="serene", text=reply))
    db.session.commit()

    return reply

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)