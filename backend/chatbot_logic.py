def detect_mood(text: str) -> str | None:
    t = text.lower()
    if any(x in t for x in ["sad", "unhappy", "down"]):
        return "sad"
    if any(x in t for x in ["angry", "mad", "annoyed"]):
        return "angry"
    if any(x in t for x in ["happy", "good", "great"]):
        return "happy"
    if any(x in t for x in ["anxious", "nervous", "scared"]):
        return "anxious"
    return None
