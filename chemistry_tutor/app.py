"""
Streamlit chat interface for the Chemistry Tutor RAG Agent.
Run with:  streamlit run chemistry_tutor/app.py
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path for Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from chemistry_tutor.config import get_config
from chemistry_tutor.ingest import build_vector_store
from chemistry_tutor.rag_chain import build_rag_chain, convert_history
from chemistry_tutor.file_utils import (
    extract_text_from_pdf,
    extract_text_from_txt,
    image_to_base64_uri,
    is_image,
    SUPPORTED_EXTS,
)
from chemistry_tutor.components.paste_button import paste_image_button

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chemistry Tutor — Class 12 CBSE",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Class 12 Chemistry Tutor")
st.caption("Powered by your NCERT notes, exemplar, and PYQs — 2027 Board Exam Edition")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    current_provider = get_config("LLM_PROVIDER", "openai").lower()
    st.caption(f"🤖 **Active Provider**: `{current_provider.upper()}`")
    REPO = "https://github.com/rajzzh-learn/Chemistry-RAG-Project/blob/main"

    st.header("🗂️ Chapters")
    st.markdown("""
**Part 1 — Physical & Inorganic**

1. The Solid State
2. Solutions
3. Electrochemistry
4. Chemical Kinetics
5. Surface Chemistry
6. General Principles & Isolation of Elements
7. The p-Block Elements
8. The d and f Block Elements
9. Coordination Compounds

---

**Part 2 — Organic**

10. Haloalkanes and Haloarenes
11. Alcohols, Phenols and Ethers
12. Aldehydes, Ketones and Carboxylic Acids
13. Amines
14. Biomolecules
15. Polymers
""")

    st.divider()

    # ── Part 3: Study Material Links ──────────────────────────────────────
    st.header("📂 Study Material")

    with st.expander("📖 Book — Part 1 (lech1dd)"):
        st.markdown(f"""
- [Chapter 1 — The Solid State]({REPO}/Book/lech1dd/lech101.pdf)
- [Chapter 2 — Solutions]({REPO}/Book/lech1dd/lech102.pdf)
- [Chapter 3 — Electrochemistry]({REPO}/Book/lech1dd/lech103.pdf)
- [Chapter 4 — Chemical Kinetics]({REPO}/Book/lech1dd/lech104.pdf)
- [Chapter 5 — Surface Chemistry]({REPO}/Book/lech1dd/lech105.pdf)
- [Appendix 1]({REPO}/Book/lech1dd/lech1a1.pdf)
- [Answers]({REPO}/Book/lech1dd/lech1an.pdf)
- [Practice Sets]({REPO}/Book/lech1dd/lech1ps.pdf)
""")

    with st.expander("📖 Book — Part 2 (lech2dd)"):
        st.markdown(f"""
- [Chapter 6 — General Principles & Isolation of Elements]({REPO}/Book/lech2dd/lech201.pdf)
- [Chapter 7 — The p-Block Elements]({REPO}/Book/lech2dd/lech202.pdf)
- [Chapter 8 — The d and f Block Elements]({REPO}/Book/lech2dd/lech203.pdf)
- [Chapter 9 — Coordination Compounds]({REPO}/Book/lech2dd/lech204.pdf)
- [Chapter 10 — Haloalkanes and Haloarenes]({REPO}/Book/lech2dd/lech205.pdf)
- [Answers]({REPO}/Book/lech2dd/lech2an.pdf)
- [Practice Sets]({REPO}/Book/lech2dd/lech2ps.pdf)
""")

    with st.expander("🔬 Exemplar"):
        st.markdown(f"""
- [Ch 1 — The Solid State]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%201%20-%20The%20Solid%20State%20(Book%20Solutions).pdf)
- [Ch 2 — Solutions]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%202%20-%20Solutions%20(Book%20Solutions).pdf)
- [Ch 3 — Electrochemistry]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%203%20-%20Electrochemistry%20(Book%20Solutions).pdf)
- [Ch 4 — Chemical Kinetics]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%204%20-%20Chemical%20Kinetics%20(Book%20Solutions).pdf)
- [Ch 5 — Surface Chemistry]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%205%20-%20Surface%20Chemistry%20(Book%20Solutions).pdf)
- [Ch 6 — Isolation of Elements]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%206%20-%20General%20Principles%20and%20Processes%20of%20Isolation%20of%20Elements%20(Book%20Solutions).pdf)
- [Ch 7 — The p-Block Elements]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%207%20-%20The%20p-Block%20Elements%20(Book%20Solutions).pdf)
- [Ch 8 — The d and f Block Elements]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%208%20-%20The%20d%20and%20f%20Block%20Elements%20(Book%20Solutions).pdf)
- [Ch 9 — Coordination Compounds]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%209%20-%20Coordination%20Compounds%20(Book%20Solutions).pdf)
- [Ch 10 — Haloalkanes and Haloarenes]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%2010%20-%20Haloalkanes%20and%20Haloarenes%20(Book%20Solutions).pdf)
- [Ch 11 — Alcohols, Phenols and Ethers]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%2011%20-%20Alcohols%2C%20Phenols%20and%20Ethers%20(Book%20Solutions).pdf)
- [Ch 12 — Aldehydes, Ketones and Carboxylic Acids]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%2012%20-%20Aldehydes%2C%20Ketones%20and%20Carboxylic%20Acids%20(Book%20Solutions).pdf)
- [Ch 13 — Amines]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%2013%20-%20Amines%20(Book%20Solutions).pdf)
- [Ch 14 — Biomolecules]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%2014%20-%20Biomolecules%20(Book%20Solutions).pdf)
- [Ch 15 — Polymers]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Chemistry%20Chapter%2015%20-%20Polymers%20(Book%20Solutions).pdf)
""")

    with st.expander("📝 Notes"):
        st.markdown(f"""
- [Ch 1 — The Solid State]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%201%20-%20Free%20PDF.pdf)
- [Ch 2 — Solutions]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%202%20-%20Free%20PDF.pdf)
- [Ch 3 — Electrochemistry]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%203%20-%20Free%20PDF.pdf)
- [Ch 4 — Chemical Kinetics]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%204%20-%20Free%20PDF.pdf)
- [Ch 5 — Surface Chemistry]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%205%20-%20Free%20PDF.pdf)
- [Ch 6 — Isolation of Elements]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%206%20-%20Free%20PDF.pdf)
- [Ch 7 — The p-Block Elements]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%207%20-%20Free%20PDF.pdf)
- [Ch 7 — The p-Block Elements (Set 2)]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%207%20-%20Free%20PDF%202.pdf)
- [Ch 8 — The d and f Block Elements]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%208%20-%20Free%20PDF.pdf)
- [Ch 8 — The d and f Block Elements (Set 2)]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%208%20-%20Free%20PDF%202.pdf)
- [Ch 9 — Coordination Compounds]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%209%20-%20Free%20PDF.pdf)
- [Ch 9 — Coordination Compounds (Set 2)]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%209%20-%20Free%20PDF%202.pdf)
- [Ch 10 — Haloalkanes and Haloarenes]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%2010%20-%20Free%20PDF.pdf)
- [Ch 10 — Haloalkanes and Haloarenes (Set 2)]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Chemistry%20Chapter%2010%20-%20Free%20PDF%202.pdf)
""")

    with st.expander("⭐ Important Questions"):
        st.markdown(f"""
- [Ch 1 — The Solid State]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%201%20-%20Free%20PDF.pdf)
- [Ch 2 — Solutions]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%202%20-%20Free%20PDF.pdf)
- [Ch 3 — Electrochemistry]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%203%20-%20Free%20PDF.pdf)
- [Ch 4 — Chemical Kinetics]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%204%20-%20Free%20PDF.pdf)
- [Ch 5 — Surface Chemistry]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%205%20-%20Free%20PDF.pdf)
- [Ch 6 — Isolation of Elements]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%206%20-%20Free%20PDF.pdf)
- [Ch 7 — The p-Block Elements]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%207%20-%20Free%20PDF.pdf)
- [Ch 7 — The p-Block Elements (Set 2)]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%207%20-%20Free%20PDF%202.pdf)
- [Ch 8 — The d and f Block Elements]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%208%20-%20Free%20PDF.pdf)
- [Ch 8 — The d and f Block Elements (Set 2)]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%208%20-%20Free%20PDF%202.pdf)
- [Ch 9 — Coordination Compounds]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%209%20-%20Free%20PDF.pdf)
- [Ch 9 — Coordination Compounds (Set 2)]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%209%20-%20Free%20PDF%202.pdf)
- [Ch 10 — Haloalkanes and Haloarenes]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%2010%20-%20Free%20PDF.pdf)
- [Ch 10 — Haloalkanes and Haloarenes (Set 2)]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Chemistry%20Chapter%2010%20-%20Free%20PDF%202.pdf)
""")

    with st.expander("🧠 Competency Based Questions"):
        st.markdown(f"""
- [Competency Based Questions — Vol 1 (2023)]({REPO}/Competency%20Based%20Questions/Chemistry_G12_Vol1_2023.pdf)
- [Competency Based Questions — Vol 2]({REPO}/Competency%20Based%20Questions/Chemistry_12Vol2.pdf)
""")

    with st.expander("📄 Previous Year Questions (PYQ)"):
        st.markdown(f"""
- [2014 Question Paper]({REPO}/PYQ/Previous%20Year%20Chemistry%20Question%20Paper%20for%20CBSE%20Class%2012%20-%202014.pdf)
- [2015 Question Paper]({REPO}/PYQ/Previous%20Year%20Chemistry%20Question%20Paper%20for%20CBSE%20Class%2012%20-%202015.pdf)
- [2016 Question Paper (Set 1C)]({REPO}/PYQ/Previous%20Year%20Chemistry%20Question%20Paper%20for%20CBSE%20Class%2012%20-%202016%20Set%201%20C.pdf)
- [2017 Question Paper]({REPO}/PYQ/CBSE%20Class%2012%202017%20Chemistry%20Question%20Paper%20-%20Free%20PDF.pdf)
- [2018 Question Paper]({REPO}/PYQ/CBSE%202018%20Chemistry%20Question%20Paper%20Class%2012%20-%20Free%20PDF.pdf)
- [2019 Question Paper]({REPO}/PYQ/CBSE%202019%20Chemistry%20Question%20Paper%20Class%2012%20-%20Free%20PDF.pdf)
- [2020 Question Paper]({REPO}/PYQ/CBSE%20Class%2012%20Chemistry%20Question%20Paper%202020.pdf)
- [2023 Question Paper]({REPO}/PYQ/2023%20-%2056-5-1_Chemistry.pdf)
- [2024 Question Paper]({REPO}/PYQ/2024%20-%2056_5_1_Chemistry.pdf)
- [2025 Question Paper with Answers]({REPO}/PYQ/CBSE%20Class%2012%20Chemistry%20Question%20Paper%202025%20Set%201%20PDF%20with%20Answers.pdf)
- [2026 Question Paper]({REPO}/PYQ/QP2026.pdf)
""")

    with st.expander("📗 NCERT Solutions"):
        st.markdown(f"""
- [Ch 1 — Solutions]({REPO}/Ncert%20Solutions/Class%2012%20Chemistry%20Chapter%201%20Solutions.pdf)
- [Ch 2 — Electrochemistry]({REPO}/Ncert%20Solutions/Class%2012%20Chemistry%20Chapter%202%20Electrochemistry.pdf)
- [Ch 3 — Chemical Kinetics]({REPO}/Ncert%20Solutions/Class%2012%20Chemistry%20Chapter%203%20Chemical%20Kinetics.pdf)
- [Ch 4 — The d and f Block Elements]({REPO}/Ncert%20Solutions/Class%2012%20Chemistry%20Chapter%204%20The%20D%20And%20F%20Block%20Elements.pdf)
- [Ch 5 — Coordination Chemistry]({REPO}/Ncert%20Solutions/Class%2012%20Chemistry%20Chapter%205%20Coordination%20Chemistry.pdf)
- [Ch 6 — Haloalkanes and Haloarenes]({REPO}/Ncert%20Solutions/Class%2012%20Chemistry%20Chapter%206%20Haloalkanes%20And%20Haloarenes.pdf)
- [Ch 7 — Alcohols, Phenols and Ethers]({REPO}/Ncert%20Solutions/Class%2012%20Chemistry%20Chapter%207%20Alcohol%20Phenol%20And%20Ether.pdf)
- [Ch 8 — Aldehydes, Ketones and Carboxylic Acids]({REPO}/Ncert%20Solutions/Class%2012%20Chemistry%20Chapter%208%20Aldehydes%20Ketones%20And%20Carboxylic%20Acids.pdf)
- [Ch 9 — Amines]({REPO}/Ncert%20Solutions/Class%2012%20Chemistry%20Chapter%209%20Amines.pdf)
- [Ch 10 — Biomolecules]({REPO}/Ncert%20Solutions/Class%2012%20Chemistry%20Chapter%2010%20Biomolecules.pdf)
""")

    with st.expander("🏫 SSM Test Question Papers"):
        st.markdown(f"""
- [SSM School Chemistry Test]({REPO}/SSM%20Questions/SSM%20School%20Chemistry%20Test.pdf)
""")

    st.divider()
    st.markdown("**💡 Try asking:**")
    st.markdown("""
- *Teach me Electrochemistry — Nernst equation*
- *Give me 5 HOTS questions on Coordination Compounds*
- *What are named reactions in Aldehydes chapter?*
- *Solve: Calculate the EMF of the cell …*
- *What were the most repeated PYQ topics in 2024?*
- *Give me all conversions for Alcohols chapter*
""")
    st.divider()
    if st.button("🔄 Rebuild Vector Store"):
        with st.spinner("Re-ingesting all PDFs … (this takes a few minutes)"):
            st.session_state.pop("rag_chain", None)
            st.session_state.pop("vector_store", None)
            build_vector_store(force_rebuild=True)
        st.success("Vector store rebuilt!")

# ── Init vector store & chain (cached in session state) ───────────────────
if "vector_store" not in st.session_state:
    with st.spinner("⚙️ Loading your study material … first load takes ~1 min"):
        st.session_state["vector_store"] = build_vector_store(force_rebuild=False)

# Re-instantiate if provider changed or not initialized
target_provider = get_config("LLM_PROVIDER", "openai").lower()
if "rag_chain" not in st.session_state or st.session_state.get("_active_provider") != target_provider:
    try:
        st.session_state["rag_chain"] = build_rag_chain(st.session_state["vector_store"])
        st.session_state["_active_provider"] = target_provider
    except Exception as e:
        st.error(
            f"⚠️ **LLM Initialization Error**: {e}\n\n"
            "👉 If using Groq, ensure `LLM_PROVIDER = \"groq\"` and `GROQ_API_KEY = \"gsk_...\"` in Streamlit Secrets.\n"
            "👉 If using OpenAI, please ensure `OPENAI_API_KEY` is added to Streamlit Secrets."
        )
        st.stop()

# ── Chat history ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Class 12 Chemistry teacher, here to help you "
                "ace your **2027 CBSE Board Exams**.\n\n"
                "Tell me which chapter or topic you want to study, "
                "or ask me to generate **named reactions**, **HOTS questions**, "
                "or **organic conversions** for any chapter! "
                "You can also **attach an image, PDF, or text file** using the 📎 button, "
                "or **paste a screenshot** with the 📋 button."
            ),
        }
    ]

# Display existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("attachment_name"):
            st.caption(f"📎 **Attached:** `{msg['attachment_name']}`")
            if msg.get("attachment_preview"):
                with st.expander("👁️ Attachment preview", expanded=False):
                    if msg.get("attachment_is_image"):
                        st.image(msg["attachment_preview"], use_container_width=True)
                    else:
                        st.text(msg["attachment_preview"][:2000])
        st.markdown(msg["content"])


# ── Helper: call vision-capable LLM directly (image path) ──────────────
def _ask_vision_llm(question: str, image_data_uri: str, context_text: str) -> str:
    """
    Send a multimodal (text + image) message to Google Gemini Flash (free tier).
    Vision is always handled by Gemini regardless of LLM_PROVIDER,
    since Groq has no vision models and OpenAI requires paid credits.
    Falls back gracefully when GOOGLE_API_KEY is not set.
    """
    import base64, re as _re
    from google import genai
    from google.genai import types

    google_key = get_config("GOOGLE_API_KEY")
    if not google_key:
        return (
            "⚠️ **Image analysis requires a `GOOGLE_API_KEY`** (free).\n\n"
            "👉 Get one at https://aistudio.google.com/app/apikey — it's free, no billing needed.\n"
            "Then add `GOOGLE_API_KEY = \"AIza...\"` to your `.env` file or Streamlit Secrets and reload the app."
        )

    # Extract raw base64 bytes from the data URI (data:<mime>;base64,<data>)
    match = _re.match(r"data:(?P<mime>[^;]+);base64,(?P<data>.+)", image_data_uri)
    if not match:
        return "⚠️ Could not parse the attached image. Please try uploading it again."
    mime_type = match.group("mime")
    image_bytes = base64.b64decode(match.group("data"))

    prompt = (
        "You are an expert Class 12 CBSE Chemistry teacher.\n\n"
        f"Context from the student's study materials:\n{context_text}\n\n"
        f"The student has attached an image and asks:\n{question}"
    )

    client = genai.Client(api_key=google_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
    )
    return response.text




# -- Input area: file uploader + paste button + chat input -----------------
ext_list = ", ".join(f".{e}" for e in sorted(SUPPORTED_EXTS))

col_upload, col_paste = st.columns([3, 1], vertical_alignment="bottom")
with col_upload:
    uploaded_file = st.file_uploader(
        f"📎 Attach a file — {ext_list}",
        type=list(SUPPORTED_EXTS),
        label_visibility="visible",
        help="Attach an image (diagram / question screenshot), PDF, or .txt file.",
    )
with col_paste:
    paste_result = paste_image_button(
        "📋 Paste image",
        background_color="#444654",
        hover_background_color="#565869",
        key="clipboard_paste",
    )

if user_input := st.chat_input("Ask your Chemistry teacher …"):
    attachment_name: str | None = None
    attachment_text: str | None = None
    attachment_image_uri: str | None = None
    attachment_preview = None
    attachment_is_image = False

    # Clipboard paste takes priority over file uploader
    if paste_result.image_data is not None:
        import io as _io
        buf = _io.BytesIO()
        paste_result.image_data.save(buf, format="PNG")
        file_bytes = buf.getvalue()
        attachment_name = "pasted-image.png"
        attachment_image_uri, _ = image_to_base64_uri(file_bytes, attachment_name)
        attachment_preview = file_bytes
        attachment_is_image = True
    elif uploaded_file is not None:
        attachment_name = uploaded_file.name
        file_bytes = uploaded_file.read()
        if is_image(attachment_name):
            attachment_image_uri, _ = image_to_base64_uri(file_bytes, attachment_name)
            attachment_preview = file_bytes
            attachment_is_image = True
        elif attachment_name.lower().endswith(".pdf"):
            attachment_text = extract_text_from_pdf(file_bytes)
            attachment_preview = attachment_text
        else:
            attachment_text = extract_text_from_txt(file_bytes)
            attachment_preview = attachment_text

    user_msg: dict = {
        "role": "user",
        "content": user_input,
        "attachment_name": attachment_name,
        "attachment_preview": attachment_preview,
        "attachment_is_image": attachment_is_image,
    }
    st.session_state["messages"].append(user_msg)

    with st.chat_message("user"):
        if attachment_name:
            st.caption(f"📎 **Attached:** `{attachment_name}`")
            if attachment_preview is not None:
                with st.expander("👁️ Attachment preview", expanded=False):
                    if attachment_is_image:
                        st.image(attachment_preview, use_container_width=True)
                    else:
                        st.text(str(attachment_preview)[:2000])
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            recent_messages = st.session_state["messages"][:-1]
            if len(recent_messages) > 4:
                recent_messages = recent_messages[-4:]
            clean_history = [{"role": m["role"], "content": m["content"]} for m in recent_messages]
            chat_history = convert_history(clean_history)

            try:
                if attachment_image_uri is not None:
                    retriever = st.session_state["vector_store"].as_retriever(
                        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20},
                    )
                    source_docs = retriever.invoke(user_input)
                    context_text = "\n\n".join(
                        f"Document {i+1} (Source: {d.metadata.get('source','?')}, "
                        f"Page: {d.metadata.get('page','?')}):\n{d.page_content}"
                        for i, d in enumerate(source_docs)
                    )
                    answer = _ask_vision_llm(user_input, attachment_image_uri, context_text)
                    sources = source_docs
                elif attachment_text is not None:
                    augmented = f"{user_input}\n\n--- Attached file: {attachment_name} ---\n{attachment_text}"
                    result = st.session_state["rag_chain"].invoke(
                        {"question": augmented, "chat_history": chat_history}
                    )
                    answer = result["answer"]
                    sources = result.get("source_documents", [])
                else:
                    result = st.session_state["rag_chain"].invoke(
                        {"question": user_input, "chat_history": chat_history}
                    )
                    answer = result["answer"]
                    sources = result.get("source_documents", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📎 Sources from your study material", expanded=False):
                        seen = set()
                        for doc in sources:
                            src = doc.metadata.get("source_label") or doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "?")
                            label = f"{src}  — page {page}"
                            if label not in seen:
                                st.markdown(f"- `{label}`")
                                seen.add(label)

                st.session_state["messages"].append({"role": "assistant", "content": answer})

            except Exception as e:
                err_msg = str(e)
                provider = get_config("LLM_PROVIDER", "openai").lower()
                if "rate_limit" in err_msg.lower() or "quota" in err_msg.lower() or "429" in err_msg:
                    if provider == "groq":
                        st.warning("⚠️ **Groq Rate Limit**: Please wait ~10-15 seconds and try again.")
                    else:
                        st.error(
                            "⚠️ **OpenAI Quota / Rate Limit Exceeded**: Check billing on "
                            "[OpenAI Billing](https://platform.openai.com/account/billing/overview) "
                            "or switch to Groq in Secrets."
                        )
                else:
                    st.error(f"⚠️ Error processing your request: {err_msg}")
