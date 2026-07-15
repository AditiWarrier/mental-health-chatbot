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
def get_session_messages(user_id):
    """Return messages from the current app chat session only."""
    return SESSION_MEMORY.get(user_id, [])


def add_session_message(user_id, role, text):
    """Store a message in current-session memory."""
    if user_id not in SESSION_MEMORY:
        SESSION_MEMORY[user_id] = []

    SESSION_MEMORY[user_id].append({
        "role": role,
        "text": text
    })

    SESSION_MEMORY[user_id] = SESSION_MEMORY[user_id][-MAX_SESSION_TURNS:]


def get_session_user_messages(user_id, limit=5):
    """Return recent user messages from the current chat session."""
    messages = get_session_messages(user_id)

    user_messages = [
        item["text"]
        for item in reversed(messages)
        if item["role"] == "user"
    ]

    return user_messages[:limit]


def get_session_last_bot_reply(user_id):
    """Return Serene's last reply from the current chat session."""
    messages = get_session_messages(user_id)

    for item in reversed(messages):
        if item["role"] == "serene":
            return item["text"]

    return ""

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
    """Compatibility helper: current chat session only."""
    return get_session_user_messages(user_id, limit=limit)

def get_last_bot_reply(user_id):
    """Compatibility helper: last Serene reply from current chat session only."""
    return get_session_last_bot_reply(user_id)

def get_recent_context(user_id, limit=8):
    """Return ordered current-session context only; never load old DB chats."""
    messages = get_session_messages(user_id)[-limit:]
    return "\n".join(f"{item['role']}: {item['text']}" for item in messages)

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
# EMOTIONAL KEYWORDS & PATTERNS
# -------------------------
FRIENDSHIP_KEYWORDS = [
    "friend", "friends", "freinds", "freind", "bestie", "best friend", "bff",
    "betray", "betrayed", "betrayal", "backstab", "backstabbed", "talk behind", "gossip",
    "secret", "secrets", "kept secret", "hide", "hiding", "lied", "lying", "lie",
    "ignore", "ignoring", "ignored", "wont talk", "wont reply", "not talking", "no reply",
    "ghost", "ghosted", "ghosting", "disappear", "disappeared", "stopped talking",
    "exclude", "excluded", "left out", "not invited", "forgot me", "didn't invite",
    "argument", "argue", "fight", "fighting", "fighting with", "yelling", "screaming",
    "lose friend", "losing friend", "scared to lose", "afraid to lose", "fear losing"
]

LONELINESS_KEYWORDS = [
    "alone", "lonely", "loneliness", "by myself", "no one", "nobody", "isolated",
    "empty", "emptiness", "hollow", "numb", "numbness", "nothing inside",
    "unwanted", "not needed", "don't belong", "outcast", "reject", "rejected",
    "abandon", "abandoned", "left behind", "everyone left", "no one stays"
]

SADNESS_KEYWORDS = [
    "sad", "sadness", "depressed", "depressing", "unhappy", "miserable",
    "cry", "crying", "cried", "tears", "tearful", "sobbing", "weep",
    "hurt", "hurting", "pain", "painful", "heartbreak", "heartbroken",
    "hopeless", "hopelessness", "giving up", "no hope", "pointless"
]

ANXIETY_KEYWORDS = [
    "anxious", "anxiety", "panic", "panicking", "panic attack",
    "overthink", "overthinking", "thoughts won't stop", "can't stop thinking",
    "worry", "worried", "worrying", "stressing", "stress", "stressed",
    "scared", "afraid", "fear", "fearful", "terrified", "frightened",
    "uncertain", "uncertainty", "unsure", "confused", "don't know", "idk",
    "nervous", "nervousness", "on edge", "tense", "restless"
]

FAMILY_KEYWORDS = [
    "family", "parent", "parents", "mom", "dad", "mother", "father",
    "sibling", "brother", "sister", "sib", "sibs",
    "fight with family", "family fight", "family argument", "family drama",
    "pressure", "pressuring", "expectation", "expectations", "demand",
    "disappoint", "disappointed", "let down", "failure", "failed"
]

ACADEMIC_KEYWORDS = [
    "school", "college", "university", "uni", "class", "classes",
    "study", "studying", "exam", "exams", "test", "tests",
    "grade", "grades", "fail", "failed", "failing", "flunk",
    "homework", "assignment", "assignments", "project", "projects",
    "deadline", "deadlines", "due date", "due dates"
]

WORK_KEYWORDS = [
    "work", "job", "boss", "coworker", "colleague", "colleagues",
    "workplace", "office", "career", "profession",
    "burnout", "burned out", "exhausted", "drained", "tired of everything",
    "overwork", "overworked", "too much work", "workload",
    "fired", "laid off", "quit", "quitting", "resign"
]

SELF_WORTH_KEYWORDS = [
    "worthless", "useless", "pointless", "good for nothing",
    "insecure", "insecurity", "not good enough", "never good enough",
    "hate myself", "hate my life", "dislike myself",
    "failure", "fail", "failed", "can't do anything", "mess everything up",
    "ugly", "fat", "stupid", "dumb", "idiot", "worthless"
]

GRIEF_KEYWORDS = [
    "grief", "grieving", "loss", "lost", "losing",
    "death", "died", "dead", "passed away", "gone",
    "miss", "miss them", "missing", "can't move on",
    "funeral", "memorial", "remember"
]

ANGER_KEYWORDS = [
    "angry", "anger", "mad", "furious", "rage", "outraged",
    "frustrated", "frustration", "annoyed", "annoying",
    "irritated", "irritation", "pissed", "pissed off"
]

DECISION_KEYWORDS = [
    "should i", "what should i do", "what do you think", "what do i do",
    "help me decide", "can't decide", "decision", "decide",
    "don't know what to do", "idk what to do", "confused", "unsure"
]

REPAIR_KEYWORDS = [
    "huh", "what", "????", "??", "???", "???",
    "you didn't understand", "you dont understand", "didnt understand",
    "that's not what i said", "thats not what i said", "not what i said",
    "why are you repeating", "stop repeating", "you're repeating",
    "i just told you", "just told you", "are you listening", "not listening"
]

# -------------------------
# RAG SYSTEM - Enhanced Retrieval
# -------------------------
KNOWLEDGE_BASE = [
    # Friendship & Trust
    "When friends betray you, it cuts deep because trust is foundational to friendship. The pain feels different from other conflicts.",
    "Secrets between friends create distance because honesty is core to real connection.",
    "Being ignored by friends can feel like a silent rejection that's hard to process.",
    "Ghosting leaves you without closure, making it harder to move forward.",
    "Being excluded hurts because it triggers fears of not belonging or mattering.",
    "Friendship arguments often reveal deeper issues that have been building up.",
    "Fear of losing friends can make you stay in situations that aren't healthy.",
    "Trust, once broken, takes consistent effort and genuine change to rebuild.",
    "Sometimes friends grow apart and that's painful but natural.",
    "Real friendship should feel safe, not constantly uncertain or anxious.",
    
    # Loneliness & Abandonment
    "Loneliness feels most painful when surrounded by people who don't understand.",
    "Feeling alone in a crowd is one of the loneliest experiences.",
    "Abandonment wounds can make it hard to trust that people will stay.",
    "Feeling unwanted often comes from past experiences of rejection.",
    "Emotional numbness can be a protective response when feelings get too overwhelming.",
    "Emptiness might mean you've been suppressing emotions for too long.",
    "Hopelessness can feel like being in a dark tunnel with no light visible.",
    "Connection with others is essential, but it starts with being kind to yourself.",
    
    # Sadness & Emotional Pain
    "Sadness that persists might need attention - your mind deserves rest and support.",
    "Crying is a natural release, not a sign of weakness.",
    "Emotional pain is just as real as physical pain and deserves care.",
    "Grief comes in waves - some days are okay, others feel impossible.",
    "It's okay to not be okay sometimes.",
    "Sadness can teach us what we value and what we need.",
    
    # Anxiety & Fear
    "Anxiety often comes from feeling overwhelmed or uncertain about the future.",
    "Overthinking is your mind trying to protect you, but it can become exhausting.",
    "Panic attacks feel scary but they do pass - you've gotten through them before.",
    "Fear of the unknown is natural, but it doesn't have to control your decisions.",
    "Anxiety can make small problems feel huge, but perspective helps.",
    "Uncertainty is uncomfortable but also means possibility exists.",
    
    # Family Conflict
    "Family conflicts hurt more because these people shaped your early understanding of love and support.",
    "Family pressure often comes from care but can feel suffocating.",
    "Setting boundaries with family is hard but sometimes necessary.",
    "You can love your family and still need space from them.",
    "Family expectations don't define your worth or your path.",
    
    # Academic Stress
    "Academic pressure can make you feel like your whole worth depends on grades.",
    "One test or assignment doesn't define your intelligence or future.",
    "College stress is common - you're not alone in feeling overwhelmed.",
    "Burnout from studying means you need rest, not more pushing.",
    "Asking for help with schoolwork is a strength, not a weakness.",
    
    # Work Stress
    "Stress about work often bleeds into personal life because it consumes so much time.",
    "Work burnout is real - it's your mind and body saying they need a break.",
    "Your job is what you do, not who you are.",
    "Workplace conflict can be especially draining because you can't always escape it.",
    "It's okay to prioritize your mental health over work demands sometimes.",
    
    # Failure & Disappointment
    "Failure feels terrible but is often how we learn and grow.",
    "Disappointment in yourself can be the hardest kind to process.",
    "One failure doesn't mean you're a failure overall.",
    "Setbacks are temporary, even when they feel permanent.",
    "You can fail at something and still be worthy and capable.",
    
    # Self-Worth & Insecurity
    "Low self-worth often comes from comparing yourself to others unfairly.",
    "Insecurity can make you interpret things negatively even when they aren't.",
    "Feeling useless doesn't make it true - feelings aren't facts.",
    "Your worth isn't determined by your productivity or achievements.",
    "Self-acceptance is a journey, not a destination you reach once.",
    
    # Grief & Loss
    "Grief has no timeline - everyone processes loss differently.",
    "Missing someone never really goes away, but the pain can soften.",
    "Grief can make you feel crazy, but your reactions are normal.",
    "It's okay to feel relief mixed with grief - emotions can be complex.",
    "Honoring someone who died doesn't mean staying stuck in pain.",
    
    # Anger & Frustration
    "Anger is often masking deeper feelings like hurt, fear, or disappointment.",
    "Frustration builds when things feel out of your control.",
    "Anger can be appropriate - it tells you something matters to you.",
    "Expressing anger healthily is different from being hurtful.",
    "Suppressing anger often leads to it coming out later in worse ways.",
    
    # Confusion & Decisions
    "Decision paralysis happens when you're overwhelmed by options and fear making wrong choice.",
    "Confusion often means you're in a transition or learning phase.",
    "Not knowing what to feel is itself a valid feeling.",
    "You don't have to have everything figured out right now.",
    "Sometimes the best decision is to wait until you feel clearer.",
    
    # General Support
    "Self-care isn't selfish - it's necessary for mental health and helping others.",
    "You don't have to go through difficult things alone.",
    "Asking for support is brave, not weak.",
    "Small steps forward still count as progress.",
    "Your feelings are valid and deserve to be acknowledged."
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
# HELPER: Check if message contains keywords from a list
# -------------------------
def contains_keywords(msg, keywords):
    """Check for whole keywords or complete multi-word phrases."""
    msg_lower = msg.lower().strip()

    for keyword in keywords:
        keyword_lower = keyword.lower().strip()

        # Multi-word phrases
        if " " in keyword_lower:
            if keyword_lower in msg_lower:
                return True
        else:
            # Single words must match as complete words
            pattern = r"\b" + re.escape(keyword_lower) + r"\b"
            if re.search(pattern, msg_lower):
                return True

    return False

# -------------------------
# HELPER: Repetition prevention
# -------------------------
def is_repetitive_response(new_response, user_id):
    """Check if new response is too similar to last bot reply"""
    last_reply = get_last_bot_reply(user_id).lower()
    new_lower = new_response.lower()
    
    # Check if identical
    if last_reply == new_lower:
        return True
    
    # Check if very similar (more than 80% overlap)
    if len(last_reply) > 0 and len(new_lower) > 0:
        similarity = SequenceMatcher(None, last_reply, new_lower).ratio()
        if similarity > 0.8:
            return True
    
    return False

# -------------------------
# HELPER: Detect conversational intent
# -------------------------
def detect_conversational_intent(response):
    """Detect the intent of a response (discovery, acknowledgment, advice, etc.)"""
    response_lower = response.lower()
    
    # Discovery questions - asking what happened
    discovery_phrases = ["what happened", "what's been going on", "what's going on", 
                         "what occurred", "can you tell me more", "what's on your mind",
                         "what's been weighing", "what about it", "what triggered"]
    
    for phrase in discovery_phrases:
        if phrase in response_lower:
            return "discovery"
    
    # Advice/suggestions
    advice_phrases = ["you could", "you might", "consider", "try to", "it might help",
                      "what would help", "what matters is", "what would feel"]
    
    for phrase in advice_phrases:
        if phrase in response_lower:
            return "advice"
    
    # Reflection/exploration
    reflection_phrases = ["how do you feel", "what do you think", "what makes you feel",
                          "do you feel more", "how are you holding", "what would make"]
    
    for phrase in reflection_phrases:
        if phrase in response_lower:
            return "reflection"
    
    return "acknowledgment"

# -------------------------
# HELPER: Detect conversation stage
# -------------------------
def detect_conversation_stage(user_id):
    """Detect conversation progression from current-session messages."""
    recent = get_session_user_messages(user_id, limit=5)

    if not recent:
        return "initial"

    current = recent[0].lower().strip()
    recent_str = " ".join(recent).lower()

    event_indicators = [
        "betray", "secret", "rumour", "rumor",
        "ghost", "ignore", "exclude", "left out",
        "argue", "fight", "fail", "failed",
        "died", "death", "pressure", "expect",
        "lied", "lie", "hid", "hide", "kept",
        "took", "stole", "said", "told",
        "did", "happened", "broke", "left"
    ]

    emotion_indicators = [
        "hurt", "angry", "sad", "upset",
        "confused", "feel", "feels", "feeling",
        "pain", "scared", "afraid", "annoyed",
        "frustrated", "lonely"
    ]

    reflection_indicators = [
        "trust", "think", "believe",
        "should i", "what should i do",
        "what do i do", "can i", "how to",
        "decide", "choice"
    ]

    has_event = any(
        indicator in recent_str
        for indicator in event_indicators
    )

    # A reasonably detailed statement about another person's action
    # also counts as an explained event.
    action_patterns = [
        r"\bmy\s+\w+\s+.+\b(hid|took|stole|lied|said|told|kept|broke|left|did)\b",
        r"\bthey\s+.+\b(hid|took|stole|lied|said|told|kept|broke|left|did)\b",
        r"\bhe\s+.+\b(hid|took|stole|lied|said|told|kept|broke|left|did)\b",
        r"\bshe\s+.+\b(hid|took|stole|lied|said|told|kept|broke|left|did)\b"
    ]

    if any(re.search(pattern, current) for pattern in action_patterns):
        has_event = True

    has_emotion = any(
        indicator in recent_str
        for indicator in emotion_indicators
    )

    has_reflection = any(
        indicator in recent_str
        for indicator in reflection_indicators
    )

    if not has_event:
        return "initial"

    if has_event and not has_emotion:
        return "event_explained"

    if has_event and has_emotion and not has_reflection:
        return "emotional_reaction"

    return "reflection"

# -------------------------
# HELPER: Extract specific details from context
# -------------------------
def extract_specific_details(user_id):
    """Extract specific details mentioned in recent conversation"""
    recent = get_session_user_messages(user_id, limit=5)
    if not recent:
        return {}
    
    recent_str = " ".join(recent).lower()
    details = {}
    
    # Check for specific friendship issues
    if "secret" in recent_str:
        details["secret"] = True
    if "rumour" in recent_str or "rumor" in recent_str:
        details["rumour"] = True
    if "betray" in recent_str:
        details["betrayal"] = True
    if "ignore" in recent_str or "ghost" in recent_str:
        details["ignored"] = True
    if "exclude" in recent_str or "left out" in recent_str:
        details["excluded"] = True
    if "argue" in recent_str or "fight" in recent_str:
        details["argument"] = True
    
    # Check for specific emotional states
    if "hurt" in recent_str:
        details["hurt"] = True
    if "angry" in recent_str or "mad" in recent_str:
        details["angry"] = True
    if "sad" in recent_str:
        details["sad"] = True
    if "confused" in recent_str or "don't know" in recent_str or "idk" in recent_str:
        details["confused"] = True
    
    # Check for specific family issues
    if "pressure" in recent_str:
        details["pressure"] = True
    if "expect" in recent_str:
        details["expectations"] = True
    
    # Check for specific academic/work fails
    if "fail" in recent_str or "failed" in recent_str:
        details["failure"] = True
    if "grade" in recent_str or "test" in recent_str or "exam" in recent_str:
        details["academic"] = True
    
    return details

# -------------------------
# HELPER: Check if short confirmation
# -------------------------
def is_short_confirmation(msg):
    """Check if message is an explicit short confirmation."""
    m = msg.lower().strip()

    short_confirmations = [
        "yes", "yeah", "yep", "exactly",
        "it does", "it hurts", "i know",
        "true", "right", "correct",
        "definitely", "absolutely", "totally",
        "for sure", "maybe", "probably"
    ]

    return m in short_confirmations

# -------------------------
# HELPER: Conversation repair
# -------------------------
def handle_conversation_repair(msg, user_id):
    """Handle messages indicating Serene misunderstood or repeated herself."""
    m = msg.lower().strip()

    if not contains_keywords(m, REPAIR_KEYWORDS):
        return None

    # Ignore the current repair message and inspect earlier user context.
    recent = get_session_user_messages(user_id, limit=6)

    previous_messages = [
        text.lower().strip()
        for text in recent
        if text.lower().strip() != m
        and not contains_keywords(text, REPAIR_KEYWORDS)
    ]

    if not previous_messages:
        return (
            "🌸 You're right — I missed what you were trying to tell me. "
            "I don't want to pretend I understood when I didn't. "
            "Could you say the last part again?"
        )

    # Most recent meaningful user message gets highest priority.
    context = " ".join(previous_messages[:3])

    # Friendship betrayal / trust
    if contains_keywords(context, FRIENDSHIP_KEYWORDS):
        if "rumour" in context or "rumor" in context:
            return (
                "🌸 You're right — you already told me. Your friends heard a rumour "
                "about you and didn't tell you, and that's left you struggling to trust them. "
                "I shouldn't have asked you to explain it again."
            )

        if "secret" in context:
            return (
                "🌸 You're right — you already told me your friends kept something important "
                "from you, and you're hurt and unsure how to feel about them now. "
                "I shouldn't have made you repeat that."
            )

        if any(word in context for word in ["betray", "backstab"]):
            return (
                "🌸 You're right — you already told me your friends betrayed you. "
                "You're trying to make sense of what that means for your trust in them. "
                "I shouldn't have asked what happened again."
            )

        if any(word in context for word in ["ignore", "ghost", "not talking"]):
            return (
                "🌸 You're right — you told me your friends haven't been talking to you. "
                "That's been making you feel alone, and I should have stayed with that instead "
                "of making you repeat yourself."
            )

        if any(word in context for word in ["exclude", "left out", "not invited"]):
            return (
                "🌸 You're right — you already told me you've been feeling left out by your friends. "
                "I should have responded to that instead of asking you to start over."
            )

        return (
            "🌸 You're right — you were already talking to me about what happened with your friends. "
            "I lost the thread of what you were saying, and I shouldn't have made you repeat yourself."
        )

    # Family
    if contains_keywords(context, FAMILY_KEYWORDS):
        return (
            "🌸 You're right — you were already telling me about what's happening with your family. "
            "I lost track of the conversation and shouldn't have asked you to explain it again."
        )

    # Academic stress
    if contains_keywords(context, ACADEMIC_KEYWORDS):
        return (
            "🌸 You're right — you were already telling me about the pressure you're dealing with "
            "around school or college. I should have stayed with what you said instead of restarting."
        )

    # Work
    if contains_keywords(context, WORK_KEYWORDS):
        return (
            "🌸 You're right — you were already telling me about what's been happening with work. "
            "I shouldn't have made you repeat yourself."
        )

    # Grief must be explicit. Do NOT infer grief from words like "remember".
    explicit_grief = [
        "grief", "grieving", "died", "death",
        "passed away", "funeral", "memorial"
    ]

    if contains_keywords(context, explicit_grief):
        return (
            "🌸 You're right — you were already telling me about the loss you're dealing with. "
            "I should have stayed with that instead of losing the thread."
        )

    # General repair
    return (
        "🌸 You're right — I missed the point of what you had just told me. "
        "I shouldn't have made you repeat yourself. Let me stay with the context you've already given me."
    )

# -------------------------
# CONVERSATION ROUTING
# -------------------------
def _topic_from_message(message):
    """Classify only what the current message explicitly supports."""
    m = message.lower().strip()

    strong_grief = [
        "grief", "grieving", "died", "death", "dead",
        "passed away", "funeral", "memorial", "bereavement"
    ]

    topic_keywords = [
        ("grief", strong_grief),
        ("friendship", FRIENDSHIP_KEYWORDS),
        ("family", FAMILY_KEYWORDS),
        ("academic", ACADEMIC_KEYWORDS),
        ("work", WORK_KEYWORDS),
        ("loneliness", LONELINESS_KEYWORDS),
        ("self_worth", SELF_WORTH_KEYWORDS),
        ("anxiety", ANXIETY_KEYWORDS),
        ("sadness", SADNESS_KEYWORDS),
        ("anger", ANGER_KEYWORDS),
    ]

    for topic, keywords in topic_keywords:
        if contains_keywords(m, keywords):
            return topic

    return None


def _active_session_topic(user_id, current_message=""):
    """Use the current message first, then the most recent explicit session topic."""
    current_topic = _topic_from_message(current_message)
    if current_topic:
        return current_topic

    recent = get_session_user_messages(user_id, limit=6)
    current_normalized = current_message.lower().strip()

    for message in recent:
        if message.lower().strip() == current_normalized:
            continue
        topic = _topic_from_message(message)
        if topic:
            return topic

    return None


def _is_advice_request(message):
    m = message.lower().strip()
    phrases = [
        "what should i do", "what do i do", "what do you think i should do",
        "how do i deal with this", "how do i deal w this", "how should i deal",
        "what can i do", "any advice", "help me decide"
    ]
    return any(phrase in m for phrase in phrases)


def _is_repair_message(message):
    m = message.lower().strip()
    phrases = [
        "i just told you", "i just told u", "just told you", "just told u",
        "why are you repeating", "stop repeating", "are you listening",
        "you didn't understand", "you dont understand", "that's not what i said",
        "thats not what i said", "not what i said", "bro what", "huh", "????"
    ]
    return m in phrases or any(phrase in m for phrase in phrases)


def _context_summary(user_id, current_message):
    previous = []
    current = current_message.lower().strip()

    for message in get_session_user_messages(user_id, limit=6):
        if message.lower().strip() != current and not _is_repair_message(message):
            previous.append(message)

    return " ".join(previous[:3]).lower()

def _wants_topic_change(message):
    m = message.lower().strip()

    phrases = [
        "change topic",
        "change the topic",
        "change subject",
        "change the subject",
        "something other than this",
        "something else",
        "talk about something else",
        "talk about sm else",
        "sm else",
        "different topic",
        "other than this",
        "get my mind off this",
        "get my mind off things"
    ]

    return any(phrase in m for phrase in phrases)


def _casual_chat_response(message):
    m = message.lower().strip()
    if "random question" in m:
          return (
            "🌸 Okay, random question 😄 "
            "If you could instantly become ridiculously good at one completely random skill, "
            "what would you choose?"
        )

    if "silly hypothetical" in m or "hypothetical" in m:
        return (
            "🌸 Okay 😄 Would you rather be able to talk to animals "
            "or understand every language in the world?"
        )

    if "movies" in m or "movie" in m:
        return "🌸 Ooh okay, movies. What's a movie you could rewatch a ridiculous number of times?"

    if "music" in m or "songs" in m or "song" in m:
        return "🌸 Okay, music 😄 What's one song you've had on repeat lately?"

    if re.search(r"\bhow are (you|u)\b", m):
        return (
            "🌸 I'm doing okay — and I'm glad you asked. "
            "We can absolutely talk about something lighter. "
            "Want a random question, a silly hypothetical, movies, music, "
            "or just normal conversation?"
        )

    if m in {
        "hi", "hey", "hello", "hii", "heyy",
        "hi serene", "hey serene", "hello serene"
    }:
        return "🌸 Hey :) I'm here. What do you feel like talking about?"

    return None
def smart_override(msg, user_id):
    """Deterministic current-message-first conversation router."""
    m = msg.lower().strip()

    # Explicit topic changes override old emotional context.
    if _wants_topic_change(msg):
        SESSION_MEMORY[user_id] = []

        return (
            "🌸 Of course. We can leave that topic here — no more exam talk. "
            "Let's talk about something completely different. "
            "Want a random question, a silly hypothetical, movies, music, "
            "or just normal conversation?"
        )

    casual_reply = _casual_chat_response(msg)

    if casual_reply:
        return casual_reply

    topic = _active_session_topic(user_id, msg)
    context = _context_summary(user_id, msg)
    last_reply = get_session_last_bot_reply(user_id).lower()

    # Repair is explicit. Ordinary words such as "what" are never repair intents.
    if _is_repair_message(msg):
        if topic == "friendship":
            return (
                "🌸 You're right — you already explained what happened with your friend. "
                "I shouldn't make you repeat it. I'm staying with what you told me: "
                "their actions hurt you and changed how you feel about the situation. "
                "What do you need most right now — help deciding what to do, or space to talk through how it felt?"
            )
        if topic == "academic":
            return (
                "🌸 You're right — you were talking about the exam and feeling underprepared. "
                "I shouldn't turn that into a failure that hasn't happened. "
                "You're worried about what might happen, and I'm staying with that."
            )
        return (
            "🌸 You're right — I lost the thread of what you said. "
            "I shouldn't invent details or make you repeat the whole story. "
            "I'm listening to the situation you've already described."
        )

    # Advice requests progress the active topic instead of restarting discovery.
    if _is_advice_request(msg):
        if topic == "friendship":
            return (
                "🌸 Based on what you've told me, I wouldn't force yourself to trust them normally right away. "
                "You could take some space, then tell them clearly what hurt you and listen to how they respond. "
                "Their response — whether they take responsibility or dismiss your feelings — can help you decide what comes next."
            )
        if topic == "academic":
            return (
                "🌸 Since the exam hasn't happened yet, I'd focus on what you can still change. "
                "Pick the highest-priority topics, study in short focused blocks, test yourself instead of only rereading, "
                "and give yourself a real break when your concentration drops. You don't need to feel fully prepared to make useful progress today."
            )
        if topic == "anxiety":
            return (
                "🌸 Try to make the next step smaller than the whole problem. "
                "Choose one thing you can control in the next 20 minutes, do that, then reassess. "
                "If your mind is racing, a short break or grounding exercise can help before you decide anything bigger."
            )
        return (
            "🌸 We can make this more manageable by separating what happened, what you can control now, "
            "and what decision actually needs to be made today. Start with the smallest useful next step rather than solving everything at once."
        )

    # Friendship: respond to the current event, not a generic keyword script.
    if topic == "friendship":
        if contains_keywords(m, ["betray", "betrayed", "betrayal", "backstab", "backstabbed"]):
            return (
                "🌸 Being betrayed by friends can shake the trust you had in the whole friendship. "
                "That hurt makes sense. What did they do that made you feel betrayed?"
            )
        if "secret" in m:
            return (
                "🌸 The fact that they kept something important from you can hurt as much as the secret itself. "
                "It makes sense that you're unsure how to feel about them now. Was it what they hid, or their choice not to tell you, that hurt more?"
            )
        if "rumour" in m or "rumor" in m:
            return (
                "🌸 Hearing that your friends knew a rumour about you and stayed silent would feel deeply personal. "
                "I can see why that damaged your trust. Are you more hurt by the rumour itself or by their silence?"
            )
        if contains_keywords(m, ["ignore", "ignoring", "ignored", "ghost", "ghosted", "not talking"]):
            return (
                "🌸 Being shut out by friends can feel lonely and confusing, especially when you don't know why. "
                "Did they suddenly stop talking to you, or was there something that happened first?"
            )
        if re.search(r"\b(hid|took|stole|lied|kept|broke|left|did)\b", m):
            return (
                "🌸 That sounds genuinely upsetting. You trusted your friend, and what they did had a real effect on you. "
                "You already told me what happened, so I won't ask you to repeat it. What part of it is bothering you most now?"
            )
        if contains_keywords(m, ["hurt", "upset", "angry", "confused"]):
            return (
                "🌸 I can hear that this is still sitting heavily with you. "
                "You don't have to decide immediately how you feel about your friend. "
                "Are you leaning more toward confronting them, or taking some distance first?"
            )
        return (
            "🌸 It sounds like something with your friend has been weighing on you. "
            "Tell me what happened, and I'll stay with the details you give me."
        )

    # Academic stress must distinguish fear of failure from an actual failure.
    if topic == "academic":
        hypothetical_failure = (
            ("what if" in m or "scared" in m or "afraid" in m or "worried" in m)
            and contains_keywords(m, ["fail", "failing"])
        )
        actual_failure = contains_keywords(m, ["failed", "flunked"])

        if hypothetical_failure:
            return (
                "🌸 You're scared you might fail — not saying that you already have. "
                "That uncertainty can make the exam feel even bigger than it is. "
                "What part feels least prepared right now: the amount left to study, remembering the material, or the time you have?"
            )
        if actual_failure:
            return (
                "🌸 Failing an exam can hurt, but one result doesn't define your ability or your future. "
                "Do you want to talk through what went wrong or focus on what you can do next?"
            )
        if contains_keywords(m, ["exam", "exams", "test", "tests"]) and contains_keywords(m, ["underprepared", "unprepared", "stressed", "stress", "worried"]):
            return (
                "🌸 A big exam can feel overwhelming when you don't feel prepared enough. "
                "The exam hasn't happened yet, so I don't want to treat your fear like a failure. "
                "How much time do you have before it?"
            )
        if "taking my mind off" in m or "take my mind off" in m:
            return (
                "🌸 That could help. A short, intentional break may give your brain room to reset instead of forcing more studying while you're overwhelmed. "
                "What usually helps you switch off for a little while?"
            )
        return (
            "🌸 Academic pressure can make one exam feel like it carries everything. "
            "I want to stay with the actual situation rather than assume the worst. What's worrying you most about it right now?"
        )

    if topic == "grief":
        return (
            "🌸 I'm sorry you're carrying this loss. Grief can change from moment to moment, "
            "and there isn't one correct way to feel. What has been hardest for you today?"
        )

    if topic == "loneliness":
        return (
            "🌸 Feeling alone can make everything else feel heavier. "
            "I'm listening. Is the loneliness coming from missing a specific person, feeling disconnected from people around you, or something else?"
        )

    if topic == "anxiety":
        if "taking my mind off" in m or "take my mind off" in m:
            return (
                "🌸 Taking your mind off it for a short while can be useful when you've been stuck in the same anxious loop. "
                "What usually helps you feel absorbed or calmer?"
            )
        return (
            "🌸 I can hear that you're feeling stressed and overwhelmed. "
            "I don't want to keep repeating the same calming question — what's driving the stress right now?"
        )

    if topic == "sadness":
        return (
            "🌸 I'm really sorry you're feeling sad. I want to understand what the sadness is connected to rather than give you a generic response. "
            "What's been weighing on you most?"
        )

    if topic == "family":
        return (
            "🌸 Family situations can carry a lot of history and pressure. "
            "What's happening with them right now?"
        )

    if topic == "work":
        return (
            "🌸 Work stress can take up a lot of mental space. "
            "What's been the most draining part lately?"
        )

    if topic == "self_worth":
        return (
            "🌸 I hear how harsh those thoughts about yourself are. "
            "I don't want to assume why you feel that way — what happened that brought those thoughts up?"
        )

    if topic == "anger":
        return (
            "🌸 You sound really frustrated. I want to understand what triggered it instead of guessing. "
            "What happened?"
        )

    return None


def fallback_response(msg, user_id):
    """Neutral fallback that never invents a topic from database history."""
    m = msg.lower().strip()
    last = get_session_last_bot_reply(user_id).lower()

    if m in {"yes", "yeah", "yep", "mhm", "mm", "exactly"}:
        topic = _active_session_topic(user_id, msg)
        if topic == "academic":
            return "🌸 I hear you. Staying with the exam situation — what feels most urgent to tackle first?"
        if topic == "friendship":
            return "🌸 I hear you. Staying with what happened with your friend — what are you feeling most strongly about it now?"
        return "🌸 I'm with you. What feels most important about this right now?"

    if any(word in m for word in ["dear lord", "ugh", "bro", "ay no", "oh my god"]):
        return (
            "🌸 I can tell you're frustrated with how this conversation is going. "
            "I won't invent a new problem. Tell me what you want me to focus on from what you've already said."
        )

    if "?" in m:
        return "🌸 I want to answer the question you're actually asking. Can you give me the one detail that matters most here?"

    if "hear you" in last or "understand" in last:
        return "🌸 I'm listening. Tell me the next part in your own words, and I'll stay with it."

    return "🌸 I'm here with you. Tell me a little more about what happened or what you're feeling right now."

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
            SESSION_MEMORY[user.id] = []
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

    add_session_message(user_id, "user", msg)

    msg_lower = msg.lower()

    if contains_keywords(msg_lower, SAFETY_KEYWORDS):
        reply = SAFETY_RESPONSE
    else:
        # 1️⃣ Smart reasoning FIRST
        # 1️⃣ Smart reasoning FIRST
        reply = smart_override(msg, user_id)
        
        # Check for repetition
        if reply and is_repetitive_response(reply, user_id):
            reply = fallback_response(msg, user_id)

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

    add_session_message(user_id, "serene", reply)

    return reply

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
