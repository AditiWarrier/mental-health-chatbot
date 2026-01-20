# app.py — Updated: robust loading, safer generation, clearer logs
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

# MODEL imports delayed so app still starts if transformers missing
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except Exception:
    GPT2LMHeadModel = None
    GPT2Tokenizer = None

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# -------------------------
# Database config (sqlite)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# -------------------------
# Models: User + Conversation
# -------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    role = db.Column(db.String(10), nullable=False)  # "user" or "serene"
    text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# -------------------------
# Load trained model (Serene)
# -------------------------
MODEL_DIR = os.path.join(BASE_DIR, "serene_model")
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

if GPT2LMHeadModel is not None:
    try:
        # Load exactly like evaluate_serene.py - no modifications!
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()
        print("✅ Loaded Serene model from", MODEL_DIR)
        print(f"✅ Device: {device}, Tokenizer vocab size: {len(tokenizer)}")
    except Exception as e:
        print("❌ Could not load Serene model:", e)
        traceback.print_exc()
        tokenizer = None
        model = None
else:
    print("❗ transformers not installed — model unavailable (app will still run).")

# -------------------------
# Safety triggers (hard-coded)
# -------------------------
SAFETY_KEYWORDS = [
    "suicide", "kill myself", "i want to die", "end my life",
    "hurt myself", "self harm", "unalive", "i want to unalive", "i dont want to live"
]

SAFETY_RESPONSE = (
    "🌸 I'm really sorry you're feeling this way. If you're in immediate danger, "
    "please call emergency services (112 in India). You don’t have to face this alone — I’m here with you."
)

# -------------------------
# Helpers: conversation buffer / prompt builder
# -------------------------
def get_recent_context(user_id, max_turns=6):
    """
    Retrieve last max_turns messages (user+serene) from DB for this user.
    Returns formatted string for prompting.
    """
    rows = (Conversation.query
            .filter_by(user_id=user_id)
            .order_by(Conversation.created_at.desc())
            .limit(max_turns)
            .all())
    # rows returned newest-first -> reverse to chronological
    rows = list(reversed(rows))
    parts = []
    for r in rows:
        if r.role == "user":
            parts.append(f"User: {r.text}")
        else:
            parts.append(f"Serene: {r.text}")
    return "\n".join(parts)

def build_prompt(user_message, user_id):
    """
    Build the prompt containing recent context plus the new user message,
    ending with 'Serene:' so model completes. Format matches training data.
    """
    context = get_recent_context(user_id, max_turns=6)
    
    # Use the same format as training data: just User: message \nSerene:
    # Don't add system prompt as model was fine-tuned on simple format
    if context:
        prompt = f"{context}\nUser: {user_message}\nSerene:"
    else:
        # First message - format exactly like training data
        prompt = f"User: {user_message}\nSerene:"
    return prompt

def clean_generated_text(text):
    """
    Clean the generated text by removing any unwanted patterns.
    Since we decode only new tokens (after prompt), we should handle:
    - Trailing "User:" or "Serene:" if model repeats structure
    - Extra whitespace
    - Leading "Serene:" if somehow present
    """
    # Remove leading "Serene:" if model generated it again (with or without whitespace)
    text = re.sub(r"^\s*Serene:\s*", "", text, flags=re.IGNORECASE)
    
    # Split on any "User:" or "Serene:" label (with or without newline) and keep only the first part
    # This handles cases like "That's hard. User: How are you?" or "Response\nSerene: More text"
    text = re.split(r"\s*User:\s*|\s*Serene:\s*", text, flags=re.IGNORECASE)[0]
    
    # Clean up whitespace but preserve sentence structure
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces/newlines with single space
    text = text.strip()
    
    # Remove any trailing incomplete patterns (shouldn't be needed after split, but safety check)
    text = re.sub(r"\s*User:\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*Serene:\s*$", "", text, flags=re.IGNORECASE)
    
    return text.strip()

# -------------------------
# Routes: auth + home
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
            return "Username already exists!", 400
        hashed_pw = generate_password_hash(password)
        user = User(username=username, password=hashed_pw)
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
        return "Invalid username or password.", 401
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# -------------------------
# Chat endpoint
# -------------------------
@app.route("/get", methods=["POST"])
def get_bot_response():
    if "user_id" not in session:
        return "Please log in to chat.", 401
    user_id = session["user_id"]
    user_message = request.form.get("msg", "").strip()
    if not user_message:
        return "Please type something 💬"

    # Save user message to DB
    conv_in = Conversation(user_id=user_id, role="user", text=user_message)
    db.session.add(conv_in)
    db.session.commit()

    lowered = user_message.lower()
    if any(k in lowered for k in SAFETY_KEYWORDS):
        # Safety response saved and returned (bypasses model)
        conv_safe = Conversation(user_id=user_id, role="serene", text=SAFETY_RESPONSE)
        db.session.add(conv_safe)
        db.session.commit()
        return SAFETY_RESPONSE

    # ALWAYS provide a response using intelligent rule-based fallback
    # This ensures the chatbot responds even if model fails
    def get_rule_based_response(msg, user_id=None):
        """Generate empathetic responses based on keywords and context - always works"""
        msg_lower = msg.lower().strip()
        
        # Handle confusion/frustration with responses
        if msg_lower in ["what", "huh", "what?", "huh?", "ugh", "um", "hm"]:
            # Get recent context to understand what they might be confused about
            try:
                recent = get_recent_context(user_id, max_turns=2) if user_id else ""
                if "should" in recent.lower() or "puppy" in recent.lower() or "pet" in recent.lower():
                    return "🌸 Getting a puppy can be a wonderful source of companionship and joy. Have you thought about what kind of care they'd need? What draws you to the idea?"
                elif "?" in recent or "question" in recent.lower():
                    return "🌸 I'm sorry if my last response wasn't clear. What question did you want me to answer?"
                else:
                    return "🌸 I'm here and listening. Can you help me understand what you're thinking about right now?"
            except:
                return "🌸 I'm here and listening. Can you help me understand what you're thinking about right now?"
        
        # Questions about getting things/pets
        if any(phrase in msg_lower for phrase in ["should i get", "what if i get", "thinking about getting", "want to get"]):
            if "puppy" in msg_lower or "dog" in msg_lower or "pet" in msg_lower:
                return "🌸 Getting a puppy can bring a lot of joy and companionship into your life. They also need consistent care and love. What draws you to the idea of having a puppy? How do you think it might help you feel?"
            elif "pet" in msg_lower:
                return "🌸 Pets can be wonderful companions and bring comfort. What kind of pet are you thinking about? What do you hope they'd bring into your life?"
            else:
                return "🌸 That sounds like something you're considering. What makes you think about that? How do you feel it might help you?"
        
        # Handle "I asked" statements - user is clarifying they asked a question
        if any(phrase in msg_lower for phrase in ["i asked", "i said", "you didn't answer", "didn't answer my question"]):
            # Check recent context for the actual question
            try:
                recent = get_recent_context(user_id, max_turns=3) if user_id else ""
                if "puppy" in recent.lower() or "dog" in recent.lower():
                    return "🌸 A puppy could be a wonderful source of comfort and companionship, especially during difficult times. They also require consistent care, time, and love. What's making you consider getting one? How do you think it might help with how you're feeling?"
                elif "should i get" in recent.lower() or "what if i get" in recent.lower():
                    return "🌸 That's a thoughtful question. Getting a puppy can bring joy and companionship, but it's also a big responsibility. What draws you to the idea? How do you feel it might help you?"
                else:
                    return "🌸 I'm sorry if I missed your question. Can you ask it again? I'm here and listening."
            except:
                return "🌸 I'm sorry if I missed your question. Can you ask it again? I'm here and listening."
        
        # Questions in general
        if msg_lower.endswith("?") or any(word in msg_lower for word in ["should i", "what if", "can i", "would it", "do you think"]):
            if "puppy" in msg_lower or "dog" in msg_lower:
                return "🌸 A puppy could be a great source of comfort and companionship. They also require time, care, and love. What's making you consider getting one? How do you think it might help with how you're feeling?"
            else:
                return "🌸 That's a thoughtful question. Can you tell me more about what's on your mind? I'm here to help you think through it."
        
        # Describing difficulty expressing feelings (check this BEFORE general emotional states)
        if any(phrase in msg_lower for phrase in ["can't describe", "cant describe", "hard to explain", "don't know how to say", "just cant describe"]):
            return "🌸 Sometimes feelings are really hard to put into words, and that's completely okay. You don't have to have the perfect words. What's the hardest part about describing what you're feeling?"
        
        # Emotional states
        if any(word in msg_lower for word in ["sad", "depressed", "down", "upset", "hurt", "depression"]):
            return "🌸 I'm really sorry you're feeling this way. I'm here with you. What's weighing on you the most right now?"
        elif any(word in msg_lower for word in ["anxious", "worried", "nervous", "stressed", "panic", "anxiety"]):
            return "🌸 That sounds really overwhelming. Let's take this gently. What's making you feel this way?"
        elif any(word in msg_lower for word in ["angry", "mad", "furious", "frustrated", "anger"]):
            return "🌸 I hear how strong those feelings are. I'm here with you. What's behind the anger?"
        elif any(word in msg_lower for word in ["tired", "exhausted", "drained", "worn", "fatigue"]):
            return "🌸 That kind of tired goes deep. I'm here. What's draining your energy?"
        elif any(word in msg_lower for word in ["lonely", "alone", "isolated", "loneliness"]):
            return "🌸 I'm here with you now. You don't have to hold this by yourself. What's making the loneliness feel stronger today?"
        elif any(word in msg_lower for word in ["overwhelmed", "too much", "can't handle", "overwhelming"]):
            return "🌸 I hear you. When everything piles up, it can feel impossible. Let's take one gentle piece at a time. What's one part you want to talk about?"
        elif any(word in msg_lower for word in ["stuck", "trapped", "can't move", "can't progress"]):
            return "🌸 Feeling stuck can be exhausting. I'm here with you. What feels like it isn't moving?"
        elif any(word in msg_lower for word in ["lost", "confused", "don't know", "uncertain"]):
            return "🌸 That sounds really heavy. Sometimes feelings are hard to put into words, and that's okay. What's the hardest part about describing it?"
        elif any(word in msg_lower for word in ["empty", "numb", "nothing", "hollow"]):
            return "🌸 Emptiness can be really draining. I'm here with you. When did this feeling start?"
        elif any(word in msg_lower for word in ["scared", "afraid", "fear", "frightened", "scary"]):
            return "🌸 Fear can be really intense. I'm here with you. What are you most afraid of right now?"
        
        # Greetings
        if any(word in msg_lower for word in ["hi", "hello", "hey"]) and len(msg.split()) <= 3:
            return "🌸 Hi there. I'm Serene, and I'm here to listen. How are you feeling today?"
        elif any(word in msg_lower for word in ["how are you", "how's it going", "what's up"]):
            return "🌸 I'm here and ready to listen. How are you doing right now? What's on your mind?"
        
        # Very short messages (1-2 words)
        if len(msg.split()) <= 2 and not msg_lower.endswith("?"):
            # Check if there's recent context suggesting confusion
            try:
                recent = get_recent_context(user_id, max_turns=2) if user_id else ""
                if "?" in recent or "question" in recent.lower():
                    return "🌸 I'm sorry if my last response didn't address your question. Can you ask it again, or tell me what you need?"
            except:
                pass
            return "🌸 I'm here for you. Can you tell me more about what's on your mind?"
        
        # Generic empathetic response - more varied
        responses = [
            "🌸 Thank you for sharing that with me. I'm here with you. What's making this feel so difficult right now?",
            "🌸 I hear you. That sounds really challenging. Can you tell me more about what you're experiencing?",
            "🌸 Thank you for opening up. I'm here with you. What part of this feels most important to talk about?",
            "🌸 I appreciate you sharing that. Let's take this gently. What's on your mind right now?",
        ]
        # Use message hash to pick a consistent but varied response
        idx = int(hashlib.md5(msg.encode()).hexdigest(), 16) % len(responses)
        return responses[idx]
    
    # Try model first, but always fall back to rule-based if it fails
    reply = None
    
    if model is not None and tokenizer is not None:
        print(f"✅ Model available - attempting generation")
        try:
            # Build prompt from recent convo + new message
            prompt = build_prompt(user_message, user_id)
            print(f"📝 Prompt: {repr(prompt[:200])}")

            # Tokenize - EXACTLY like evaluate_serene.py
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # Limit prompt length
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            input_len = inputs["input_ids"].shape[1]
            print(f"📏 Tokenized length: {input_len}")

            # Generate - Match evaluate script but ensure max_length >= input_len
            max_gen_length = max(120, input_len + 80)  # At least 80 new tokens, or 120 total (whichever is larger)
            print(f"🔧 Generating with max_length={max_gen_length}")
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_length=max_gen_length,  # Ensure it's >= input length
                    temperature=0.7,  # Slightly higher than 0.6 for more variety
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode - EXACTLY like evaluate_serene.py
            text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"🔍 Full generated ({len(text)} chars): {repr(text[:300])}")

            # Extract reply - EXACTLY like evaluate_serene.py
            if "Serene:" in text:
                reply = text.split("Serene:", 1)[1].strip()
                print(f"📌 Extracted after 'Serene:': {repr(reply[:200])}")
            else:
                # If no "Serene:" marker, the model might have generated something unexpected
                # Try to extract just the new part after the prompt
                generated_tokens = output_ids[0][input_len:]
                reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                print(f"📌 Extracted new tokens only: {repr(reply[:200])}")
            
            # Clean up: remove any trailing "User:" or new "Serene:" if model continues
            reply = re.split(r"\s*User:\s*|\s*Serene:\s*", reply, flags=re.IGNORECASE)[0].strip()
            reply = re.sub(r"\s+", " ", reply).strip()
            
            print(f"✨ Final cleaned reply ({len(reply)} chars): {repr(reply[:200])}")
            
            # Validation: ensure reply is meaningful (at least 3 characters and not just labels)
            reply_lower = reply.lower().strip()
            if not reply or len(reply) < 3 or reply_lower in ["serene:", "serene", "user:", "user", ""]:
                print("⚠️ Reply too short or invalid, will use fallback")
                reply = None
            else:
                print(f"✅ Model generated valid response")

        except Exception as e:
            # Print detailed traceback to server console for debugging
            print(f"❌ Generation error for message '{user_message}': {e}")
            print("Full traceback:")
            traceback.print_exc()
            print("⚠️ Falling back to rule-based response")
            reply = None
    
    # Use rule-based response (always works, even if model failed or unavailable)
    if reply is None or not reply or len(reply.split()) <= 1:
        reply = get_rule_based_response(user_message, user_id)
        print(f"🌸 Using rule-based response: {repr(reply[:100])}")
    
    # Save bot reply (always succeeds now)
    try:
        conv_out = Conversation(user_id=user_id, role="serene", text=reply)
        db.session.add(conv_out)
        db.session.commit()
    except Exception as db_error:
        print(f"⚠️ Could not save reply to DB: {db_error}")
    
    return reply

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
