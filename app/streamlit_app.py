# app/streamlit_app.py

# --- ensure project root is importable (fixes ModuleNotFoundError when Streamlit runs) ---
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ---------------------------------------------------------------------

import traceback
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# app internals
from app.db import init_db, SessionLocal, User, Conversation, Message
from app.sentiment import analyze_sentiment
from app.api_client import ask_mental, ask_legal

# rewards optional
try:
    from app.rewards import compute_session_reward
    HAS_REWARDS = True
except Exception:
    HAS_REWARDS = False

# optional retrieval (ingest + query)
try:
    from app.retrieval import ingest_folder, query as kb_query
    HAS_RETRIEVAL = True
except Exception:
    ingest_folder = None
    kb_query = None
    HAS_RETRIEVAL = False

# ensure KB folder
KB_FOLDER = os.path.join("data", "kb_docs")
os.makedirs(KB_FOLDER, exist_ok=True)

# initialize DB
init_db()

# Force single model only
SELECTED_MODEL = "models/gemini-2.5-flash"

# Streamlit page setup
st.set_page_config(page_title="AI Smart Companion", layout="wide")
st.title("AI Smart Companion")

# Sidebar: mode, user, KB upload, controls
section = st.sidebar.radio("Mode", ["Mental Health", "Legal Assistance"], index=0)
username = st.sidebar.text_input("Your name (optional)")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model (fixed)**")
st.sidebar.info(f"This app uses a single model: `{SELECTED_MODEL}`")

st.sidebar.markdown("---")
st.sidebar.markdown("### Knowledge base (legal)")
uploaded_files = st.sidebar.file_uploader("Upload PDFs / TXT", type=["pdf", "txt"], accept_multiple_files=True, key="upload_kb")

if uploaded_files and st.sidebar.button("Save files to KB", key="save_kb"):
    saved = []
    for f in uploaded_files:
        target = os.path.join(KB_FOLDER, f.name)
        base, ext = os.path.splitext(target)
        i = 1
        while os.path.exists(target):
            target = f"{base}_{i}{ext}"
            i += 1
        with open(target, "wb") as out:
            out.write(f.getbuffer())
        saved.append(os.path.basename(target))
    st.sidebar.success(f"Saved {len(saved)} files.")
    if saved:
        st.sidebar.write(saved)

if HAS_RETRIEVAL:
    if st.sidebar.button("Ingest KB now", key="ingest_kb"):
        try:
            with st.spinner("Ingesting documents and building embeddings..."):
                count = ingest_folder(KB_FOLDER)
            st.sidebar.success(f"Ingested {count} chunks.")
        except Exception:
            st.sidebar.error("Ingest failed — see traceback.")
            st.sidebar.code(traceback.format_exc())
else:
    st.sidebar.info("Retrieval (KB) not available on this deployment.")

st.sidebar.markdown("---")
if st.sidebar.button("Start New Chat", key="start_new_chat"):
    st.session_state.conversation_id = None
    st.experimental_rerun()

# End session reward
if st.sidebar.button("End Session & Get Reward", key="end_session_btn"):
    st.session_state._end_session_requested = True
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: Use 'Start New Chat' to begin a fresh conversation.")

# Conversation state
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# Chat input area
st.markdown("---")
st.header("Chat with your assistant")
user_msg = st.text_area("Your message...", height=150, key="main_input")
send_clicked = st.button("Send", key="send_main")

# Send logic
if send_clicked and user_msg.strip():
    db = SessionLocal()

    # create/find user
    user = None
    if username:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            user = User(username=username)
            db.add(user)
            db.commit()
            db.refresh(user)

    # create/find conversation
    conv = None
    if st.session_state.conversation_id:
        conv = db.query(Conversation).filter(Conversation.id == st.session_state.conversation_id).first()

    if not conv:
        conv = Conversation(
            user_id=(user.id if user else None),
            section=("mental" if section == "Mental Health" else "legal")
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        st.session_state.conversation_id = conv.id

    # Save user message ONCE
    try:
        user_message_obj = Message(conversation_id=conv.id, sender="user", text=user_msg, sentiment=None)
        db.add(user_message_obj)
        db.commit()
        db.refresh(user_message_obj)
    except Exception as e:
        db.rollback()
        st.error("Failed to save your message to the database.")
        st.code(str(e))
        # still continue (we'll use user_msg variable for generation)

    # Sentiment if mental
    sentiment = None
    try:
        if section == "Mental Health":
            sentiment = analyze_sentiment(user_msg)
    except Exception:
        sentiment = None

    # Generate reply and save bot reply reliably
    with st.spinner("Assistant is typing..."):
        try:
            if section == "Mental Health":
                reply = ask_mental(user_msg, sentiment, model_name=SELECTED_MODEL)
                sources = []
            else:
                retrieved = []
                if kb_query:
                    try:
                        retrieved = kb_query(user_msg, k=5)
                    except Exception:
                        retrieved = []
                reply, sources = ask_legal(user_msg, retrieved_passages=retrieved, model_name=SELECTED_MODEL)
        except Exception as e:
            print("Generation error:", e)
            reply = "Sorry — I couldn't generate a response right now. Please try again later."
            sources = []

        # Save assistant reply reliably
        try:
            bot_message_obj = Message(
                conversation_id=conv.id,
                sender="bot",
                text=reply,
                sentiment=(sentiment if section == "Mental Health" else None)
            )
            db.add(bot_message_obj)
            db.commit()
            db.refresh(bot_message_obj)
        except Exception as e:
            db.rollback()
            st.error("Failed to save assistant message to the database (check logs).")
            st.code(str(e))

        # If user requested end-of-session reward, compute and show it
        if st.session_state.get("_end_session_requested") and HAS_REWARDS:
            try:
                all_msgs = db.query(Message).filter(Message.conversation_id == conv.id).order_by(Message.id.asc()).all()
                serial_msgs = [{"sender": m.sender, "text": m.text} for m in all_msgs]
                reward = compute_session_reward(conv.id, serial_msgs)
                try:
                    st.balloons()
                except Exception:
                    pass
                st.success(f"Session Reward: {reward['badge']} — {reward['score']} points")
                st.markdown(f"**{reward['note']}**")
                st.markdown(f"_Messages this session: {reward.get('message_count', 0)}_")
                st.json({"first_sentiment": reward.get("first_sentiment"), "last_sentiment": reward.get("last_sentiment")})
            except Exception:
                st.error("Failed to compute session reward.")
                st.code(traceback.format_exc())
            finally:
                st.session_state._end_session_requested = False

    # show assistant reply
    st.markdown("### Assistant:")
    st.write(reply)

    # show sources (for legal)
    if section == "Legal Assistance" and sources:
        st.markdown("**Sources (top results):**")
        for i, s in enumerate(sources):
            src_name = s.get("source") or f"doc_{i}"
            score = s.get("score", 0)
            snippet = s.get("text", "")[:500]
            col1, col2 = st.columns([8, 2])
            with col1:
                st.markdown(f"**{src_name}** — _score: {score:.3f}_")
                st.write(snippet + ("..." if len(s.get("text", "")) > 500 else ""))
            with col2:
                kb_path = os.path.join("data", "kb_docs", src_name)
                if os.path.exists(kb_path):
                    try:
                        with open(kb_path, "rb") as f:
                            st.download_button(label="Download", data=f, file_name=src_name, key=f"dl_{src_name}_{i}")
                    except Exception:
                        st.write("Download error")
                else:
                    st.write("No file")

# Conversation history rendering (ordered by id)
if st.session_state.conversation_id:
    st.markdown("---")
    st.subheader("Conversation history")
    db = SessionLocal()
    conv = db.query(Conversation).filter(Conversation.id == st.session_state.conversation_id).first()
    if conv:
        msgs = db.query(Message).filter(Message.conversation_id == conv.id).order_by(Message.id.asc()).all()
        for m in msgs:
            # user bubble (lighter green, dark text)
            if m.sender == "user":
                st.markdown(
                    f"""
                    <div style="
                        background:#AFE1AF;
                        color:#000;
                        padding:12px;
                        border-radius:12px;
                        max-width:75%;
                        margin-left:auto;
                        margin-bottom:8px;
                        font-size:16px;
                        line-height:1.5;
                        white-space:pre-wrap;
                    ">
                        {m.text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # bot bubble (neutral grey)
                st.markdown(
                    f"""
                    <div style="
                        background:#E7E7E7;
                        color:#000;
                        padding:12px;
                        border-radius:12px;
                        max-width:75%;
                        margin-right:auto;
                        margin-bottom:8px;
                        font-size:16px;
                        line-height:1.5;
                        white-space:pre-wrap;
                    ">
                        {m.text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
