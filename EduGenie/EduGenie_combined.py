import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LOCAL_EXPLAINER_ENABLED = os.getenv("LOCAL_EXPLAINER_ENABLED", "false").lower() == "true"
LOCAL_EXPLAINER_MODEL = os.getenv(
    "LOCAL_EXPLAINER_MODEL", "MBZUAI/LaMini-Flan-T5-783M"
)

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "12000"))


# ===== DATA MODELS =====
from pydantic import BaseModel, Field, field_validator

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=12000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Input cannot be empty.")
        return value

class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    explanation: str

class QuizResponse(BaseModel):
    questions: list[QuizQuestion] = Field(min_length=3, max_length=3)


# ===== GEMINI_CLIENT.PY =====
from google import genai
from google.genai import types


_client = None


def get_client():
    global _client

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add your Google Gemini API key."
        )

    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)

    return _client


def generate_text(prompt: str, *, system_instruction: str | None = None) -> str:
    if not prompt.strip():
        raise ValueError("Input cannot be empty.")

    prompt = prompt[:MAX_INPUT_CHARS]

    config_kwargs = {
        "temperature": 0.3,
        "max_output_tokens": 1500,
    }

    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    return text.strip()


def generate_json(prompt: str, schema):
    prompt = prompt[:MAX_INPUT_CHARS]

    response = get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1800,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    if getattr(response, "parsed", None) is not None:
        return response.parsed

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned no structured response.")

    return text


# ===== QNA.PY =====

SYSTEM = """You are EduGenie, a student-friendly educational assistant.
Answer accurately and concisely. Explain unfamiliar terms in simple language.
If the question is ambiguous, state the assumption you are making.
Do not pretend to know information you are unsure about."""


def answer_question(question: str) -> str:
    prompt = f"""Answer the student's question.

Question:
{question}

Give a clear answer first, followed by a short explanation or example when useful."""
    return generate_text(prompt, system_instruction=SYSTEM)


# ===== EXPLANATION_MODULE.PY =====

_pipeline = None


def _get_local_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "text2text-generation",
            model=LOCAL_EXPLAINER_MODEL,
            max_new_tokens=220,
        )
    return _pipeline


def _local_explain(topic: str) -> str:
    prompt = (
        "Explain the following topic to a beginner in simple, concise language. "
        "Use a small example if helpful.\n\nTopic: " + topic
    )
    result = _get_local_pipeline()(prompt)
    return result[0]["generated_text"].strip()


def explain_topic(topic: str) -> str:
    if LOCAL_EXPLAINER_ENABLED:
        try:
            return _local_explain(topic)
        except Exception:
            # If the local model is unavailable, keep the application usable.
            pass

    return generate_text(
        f"""Explain this topic to a beginner: {topic}

Use:
- a one-sentence definition,
- 2 to 4 simple points,
- one small example,
- one real-world use.

Avoid unnecessary jargon."""
    )


# ===== QUIZ_MODULE.PY =====


def generate_quiz(text: str) -> QuizResponse:
    prompt = f"""Create exactly 3 multiple-choice questions from the educational
content below.

Rules:
- Each question has exactly 4 options.
- Exactly one option is correct.
- correct_answer must exactly match one of the four option strings.
- Add a short explanation for the correct answer.
- Questions must test understanding of the supplied content.
- Do not add facts unrelated to the content.

Content:
{text}"""

    result = generate_json(prompt, QuizResponse)

    if isinstance(result, QuizResponse):
        return result

    if isinstance(result, dict):
        return QuizResponse.model_validate(result)

    return QuizResponse.model_validate_json(result)


# ===== SUMMARY_MODULE.PY =====


def summarize_text(text: str) -> str:
    prompt = f"""Summarize the educational text below for a student.

Requirements:
- Keep the main facts and important relationships.
- Remove repetition and unnecessary detail.
- Use simple language.
- Prefer short paragraphs and bullet points when appropriate.
- Do not add facts that are not present in the source.

Text:
{text}"""
    return generate_text(prompt)


# ===== LEARNING_PATH.PY =====


def get_learning_recommendations(topic: str) -> str:
    prompt = f"""Create a personalized learning path for this topic: {topic}

Structure it from beginner to advanced.
For each stage include:
1. What to learn
2. Why it matters
3. A realistic suggested timeline
4. Practice ideas

Finish with useful resource types such as documentation, videos, articles,
or books. Do not invent exact URLs unless you are certain they exist.
Keep the plan practical for a student."""
    return generate_text(prompt)


# ===== FASTAPI APPLICATION =====
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI(
    title="EduGenie",
    description="Google Gemini powered learning assistant",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "EduGenie"}


@app.post("/qa")
async def qa(request: TextRequest):
    try:
        return {"result": answer_question(request.text)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/explain")
async def explain(request: TextRequest):
    try:
        return {"result": explain_topic(request.text)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/quiz")
async def quiz(request: TextRequest):
    try:
        return generate_quiz(request.text).model_dump()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/summarize")
async def summarize(request: TextRequest):
    try:
        return {"result": summarize_text(request.text)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/learn/recommendations")
async def recommendations(request: TextRequest):
    try:
        return {"result": get_learning_recommendations(request.text)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
