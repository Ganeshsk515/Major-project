# app/rewards.py
import os
import json
from datetime import datetime
from typing import List, Dict, Any

# Attempt to import the project's sentiment analyzer; fallback if missing
try:
    from app.sentiment import analyze_sentiment
except Exception:
    def analyze_sentiment(text: str) -> Dict[str, Any]:
        pos = ["good", "better", "happy", "relieved", "okay", "great", "hope", "improved"]
        neg = ["sad", "angry", "depressed", "bad", "worse", "suicid", "hate", "anxious"]
        t = (text or "").lower()
        score = 0.0
        for w in pos:
            if w in t: score += 0.4
        for w in neg:
            if w in t: score -= 0.6
        if score > 1.0: score = 1.0
        if score < -1.0: score = -1.0
        label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
        return {"score": score, "label": label}

# Persistence
DATA_DIR = os.path.join("data")
REWARDS_PATH = os.path.join(DATA_DIR, "rewards.json")
os.makedirs(DATA_DIR, exist_ok=True)

def _sentiment_to_numeric(s: Dict[str, Any]) -> float:
    if s is None:
        return 0.0
    if isinstance(s, dict):
        if "score" in s and isinstance(s["score"], (int, float)):
            val = float(s["score"])
            if "label" in s:
                lab = str(s["label"]).lower()
                if "neg" in lab:
                    return -abs(val)
                elif "pos" in lab:
                    return abs(val)
            # try map 0..1 to -1..1 if seems necessary
            if -1.0 <= val <= 1.0:
                return val
            return (val * 2.0) - 1.0
    return 0.0

def _persist_reward(rec: Dict[str, Any]):
    try:
        data = []
        if os.path.exists(REWARDS_PATH):
            with open(REWARDS_PATH, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []
        data.append(rec)
        tmp = REWARDS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, REWARDS_PATH)
    except Exception as e:
        print("Failed to persist reward:", e)

def compute_session_reward(conv_id: int, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute a reward for the session described by `messages`.
    messages: list of dicts with keys 'sender' and 'text'.
    Returns a record with score, badge, note, and stats. Persists to data/rewards.json.
    """
    user_texts = [m["text"] for m in messages if m.get("sender") == "user"]
    if not user_texts:
        score = 5
        badge = "Keep Going"
        note = "You started a session — every step counts. Try chatting a bit to earn rewards!"
        rec = {"conv_id": conv_id, "score": score, "badge": badge, "note": note, "timestamp": datetime.utcnow().isoformat()}
        _persist_reward(rec)
        return rec

    first_text = user_texts[0]
    last_text = user_texts[-1]

    s_first = analyze_sentiment(first_text)
    s_last = analyze_sentiment(last_text)

    v_first = _sentiment_to_numeric(s_first)
    v_last = _sentiment_to_numeric(s_last)

    msg_count = len(user_texts)
    improvement = v_last - v_first

    positivity_bonus = max(0, v_last) * 15
    engagement_bonus = min(20, msg_count * 2)
    improvement_bonus = improvement * 25

    raw_score = 40 + improvement_bonus + engagement_bonus + positivity_bonus
    score = int(max(0, min(100, round(raw_score))))

    if score >= 80:
        badge = "Gold 🌟"
        note = "Amazing progress — keep it up! You're doing great."
    elif score >= 60:
        badge = "Silver 🥈"
        note = "Great session — you're on the right track!"
    elif score >= 40:
        badge = "Bronze 🥉"
        note = "Nice work — small steps add up. Try again soon!"
    else:
        badge = "Keep Going 💪"
        note = "Every session helps. Try a short breathing exercise and come back later."

    rec = {
        "conv_id": conv_id,
        "score": score,
        "badge": badge,
        "note": note,
        "first_sentiment": {"value": v_first, "raw": s_first},
        "last_sentiment": {"value": v_last, "raw": s_last},
        "message_count": msg_count,
        "timestamp": datetime.utcnow().isoformat()
    }

    _persist_reward(rec)
    return rec
