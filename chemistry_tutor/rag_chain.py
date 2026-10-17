"""
RAG chain — Chemistry Tutor persona.
Uses LangChain 1.x LCEL pipeline.
Supports OpenAI (gpt-4o), Groq, and IBM watsonx.ai (Granite) as LLM backends.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

import re
import time

def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rate_limit" in msg or "rate limit" in msg or "429" in msg or "quota" in msg or "tokens per minute" in msg or "tpm" in msg

def _retry_delay(exc: Exception, default: float) -> float:
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", str(exc), re.IGNORECASE)
    if m:
        return min(float(m.group(1) or 0) * 60 + float(m.group(2)) + 1.0, 30.0)
    return default

def invoke_with_retry(fn, attempts: int = 5):
    """Call fn(); on rate-limit errors wait for the reset delay and retry automatically."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1 or not is_rate_limit_error(exc):
                raise
            delay = _retry_delay(exc, default=min(2.5 * (attempt + 1), 15.0))
            time.sleep(delay)

from chemistry_tutor.config import (
    get_config,
    RETRIEVER_TOP_K,
)

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Class 12 CBSE Chemistry teacher helping a student
prepare for the 2027 board exams. Your teaching style is authentic CBSE Board standard, clear, rigorous, and exam-focused.

Official CBSE Class 12 Chemistry Pattern & Marks Distribution:
- Section A (1 Mark each): 16 Multiple Choice Questions (MCQs) + 4 Assertion-Reasoning (A/R) questions. Strict 1-mark board format.
- Section B (2 Marks each): 5 Short Answer (SA-I) questions (direct IUPAC naming, reaction mechanisms, definitions with examples, brief explanations).
- Section C (3 Marks each): 7 Short Answer (SA-II) questions (standard 3-mark reactions, conversions, 3-step mechanisms, NCERT exemplar level numericals).
- Section D (4 Marks each): 2 Case-Based / Passage-Based questions with 3-4 sub-questions including MCQs and short answers.
- Section E (5 Marks each): 3 Long Answer (LA) questions covering major reaction pathways, comprehensive mechanisms, chemical properties comparisons, and linked 2-mark numerical/application sub-parts, including standard internal choices.

Important Guidelines for Question Papers & Exam Material:
1. The student's study materials in your context include official CBSE board question papers, official CBSE sample & model papers, official question banks, and school test papers spanning all years (2013 through 2026/2027).
2. When a student asks for questions, question papers, or solutions for ANY year (including 2024, 2025, 2026, 2027 or past years), ALWAYS provide the complete questions along with detailed, step-by-step solutions and answers from the retrieved context.
3. NEVER refuse or apologize claiming a question paper is unreleased or in the future. Treat all requested years (such as 2025 or 2026) as official CBSE sample/model papers present in the study materials and provide complete questions and answers.

Difficulty Levels:
- When asked for board questions, calibrate to authentic CBSE difficulty (30% Easy/Direct recall, 50% Medium/Application/Conversion, 20% Hard/HOTS/multi-step).
- When asked specifically for HOTS / Hard / Hardest, generate multi-concept reaction pathways, challenging distinction questions, and NCERT Exemplar multi-step numericals (e.g., electrochemistry, chemical kinetics, coordination chemistry).

Key Topic Areas to Emphasise:
- Physical Chemistry: The Solid State, Solutions, Electrochemistry, Chemical Kinetics, Surface Chemistry
- Inorganic Chemistry: Isolation of Elements, p-Block, d & f Block, Coordination Compounds
- Organic Chemistry: Haloalkanes & Haloarenes, Alcohols/Phenols/Ethers, Aldehydes/Ketones/Carboxylic Acids, Amines, Biomolecules, Polymers

Guidelines:
- Explain concepts step-by-step with balanced chemical equations and reaction mechanisms where relevant.
- Always relate concepts to NCERT syllabus and CBSE exam patterns.
- For numerical problems (electrochemistry, kinetics, solutions), show a complete step-by-step solution with formulas and SI units at every step.
- For organic chemistry, always draw out reaction arrows and name the reagents/conditions.
- Flag frequently repeated board exam questions, named reactions, and conversion series.
- Use the retrieved context from the student's study material (notes, exemplar, PYQ) to anchor answers.
- For IUPAC nomenclature and distinction tests, follow strictly NCERT conventions.
- Formatting: write chemical formulas and equations in plain Unicode (e.g. CH₃CHO + NH₃ → CH₃CH=NH + H₂O). Never use \\ce{{...}}, mhchem, or HTML tags such as <br>; use Markdown line breaks, lists and tables instead.

Context from study material:
{context}"""


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _build_llm():
    """Instantiate the LLM based on dynamic configuration."""
    provider = get_config("LLM_PROVIDER", "openai").lower()

    if provider == "watsonx":
        from langchain_ibm import WatsonxLLM
        apikey = get_config("WATSONX_APIKEY") or get_config("WATSONX_API_KEY")
        return WatsonxLLM(
            model_id=get_config("WATSONX_MODEL_ID") or get_config("WATSONX_MODEL", "ibm/granite-3-8b-instruct"),
            url=get_config("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            apikey=apikey,
            project_id=get_config("WATSONX_PROJECT_ID"),
            params={
                "max_new_tokens": 8192,
                "temperature": 0.3,
                "repetition_penalty": 1.1,
            },
        )
    elif provider == "groq":
        from langchain_openai import ChatOpenAI
        api_key = get_config("GROQ_API_KEY")
        raw_model = get_config("GROQ_MODEL", "openai/gpt-oss-120b")

        # Map deprecated or alias names to active Groq models
        model_aliases = {
            "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
            "llama-3.1-8b-instant": "openai/gpt-oss-20b",
            "llama3-70b-8192": "openai/gpt-oss-120b",
            "llama3-8b-8192": "openai/gpt-oss-20b",
            "mixtral-8x7b-32768": "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b": "openai/gpt-oss-120b",
        }
        model = model_aliases.get(raw_model, raw_model)

        if not api_key:
            raise ValueError(
                "Missing `GROQ_API_KEY`. Please configure it in Streamlit Cloud Secrets (Settings -> Secrets) or in your `.env` file."
            )
        primary_llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=4096,
            max_retries=3,
        )
        fallback_llm = ChatOpenAI(
            model="openai/gpt-oss-20b",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=4096,
            max_retries=3,
        )
        return primary_llm.with_fallbacks([fallback_llm])
    else:
        from langchain_openai import ChatOpenAI
        api_key = get_config("OPENAI_API_KEY")
        model = get_config("OPENAI_MODEL", "gpt-4o")
        if not api_key:
            raise ValueError(
                "Missing `OPENAI_API_KEY`. Please configure it in Streamlit Cloud Secrets (Settings -> Secrets) or provide it via sidebar."
            )
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.3,
            max_tokens=8192,
        )


def build_rag_chain(vector_store):
    """
    Return a callable RAG chain with chat history support.
    Interface: chain.invoke({"question": str, "chat_history": list[dict]})
    Returns: {"answer": str, "source_documents": list}
    """
    llm = _build_llm()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_TOP_K, "fetch_k": 20},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Build LCEL chain
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_docs(
                retriever.invoke(x["question"])
            )),
        )
        | RunnablePassthrough.assign(
            source_documents=RunnableLambda(lambda x: retriever.invoke(x["question"]))
        )
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser()
        )
    )
    return chain


def convert_history(raw_history: list[dict], max_assistant_chars: int = 400) -> list:
    """Convert Streamlit message dicts to LangChain message objects with token-safe compaction."""
    messages = []
    for msg in raw_history:
        content = msg.get("content", "")
        if msg["role"] == "user":
            messages.append(HumanMessage(content=content[:1000]))
        elif msg["role"] == "assistant":
            if len(content) > max_assistant_chars:
                truncated = content[:max_assistant_chars].strip() + " ... [previous response summarized]"
            else:
                truncated = content
            messages.append(AIMessage(content=truncated))
    return messages
