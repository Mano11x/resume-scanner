"""
╔══════════════════════════════════════════╗
║  RESUMAI — app.py                        ║
║  Flask backend for Resume Intelligence   ║
╚══════════════════════════════════════════╝

Usage:
  pip install flask anthropic pdfminer.six python-docx
  python app.py

Then open:  http://localhost:5000
"""

import os
import io
import json
import re
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
import anthropic

# PDF extraction
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
    from pdfminer.pdfpage import PDFPage
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[WARN] pdfminer.six not installed — PDF text extraction disabled.")

# DOCX extraction
try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("[WARN] python-docx not installed — DOCX extraction disabled.")


# ─── APP SETUP ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")

# Anthropic client — reads ANTHROPIC_API_KEY from env automatically
# When running from claude.ai artifacts context the key is injected by the platform
client = anthropic.Anthropic()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}
MAX_FILE_SIZE_MB   = 10
CLAUDE_MODEL       = "claude-sonnet-4-5"


# ─── HELPERS ───────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from uploaded resume file."""
    ext = Path(filename).suffix.lower()

    # ── Plain text ──
    if ext == ".txt":
        return file_bytes.decode("utf-8", errors="replace")

    # ── PDF ──
    if ext == ".pdf":
        if not PDF_SUPPORT:
            return ""
        try:
            text = pdf_extract_text(io.BytesIO(file_bytes))
            return (text or "").strip()
        except Exception as e:
            print(f"[WARN] PDF extraction error: {e}")
            return ""

    # ── DOCX ──
    if ext in (".docx", ".doc"):
        if not DOCX_SUPPORT:
            return ""
        try:
            doc  = DocxDocument(io.BytesIO(file_bytes))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return text.strip()
        except Exception as e:
            print(f"[WARN] DOCX extraction error: {e}")
            return ""

    return ""


def parse_json_response(raw: str) -> dict:
    """Safely parse JSON from Claude response, stripping markdown fences."""
    clean = raw.strip()
    clean = re.sub(r"^```json\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^```\s*",     "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"```\s*$",     "", clean)
    clean = clean.strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Try to find first { ... } block
        m = re.search(r"\{[\s\S]*\}", clean)
        if m:
            return json.loads(m.group())
        raise ValueError("Could not parse JSON from AI response.")


ANALYSIS_SYSTEM_PROMPT = """You are a world-class resume analyst and ATS optimization specialist.
Return ONLY a raw JSON object — absolutely no markdown fences, no backticks, no text outside the JSON.

Required JSON schema:
{
  "overallScore":  <integer 0-100>,
  "atsScore":      <integer 0-100>,
  "impactScore":   <integer 0-100>,
  "clarityScore":  <integer 0-100>,
  "improvements": [
    { "severity": "high"|"medium"|"low", "title": "string", "detail": "2-3 sentence actionable advice" }
  ],
  "strengths":       ["string", ...],
  "weaknesses":      ["string", ...],
  "keywordsFound":   ["string", ...],
  "keywordsMissing": ["string", ...],
  "detectedRole":    "string",
  "summary":         "2-3 sentence verdict",
  "flowSteps": [
    { "step": 1, "title": "string", "desc": "string", "priority": "high"|"medium"|"low", "tags": ["string"] }
  ],
  "sampleContent": {
    "summary":      "Compelling 3-sentence professional summary",
    "experience":   "3 strong bullet points using XYZ formula with metrics",
    "skills":       "12 relevant technical and soft skills as natural sentences",
    "achievements": "2 specific quantified achievement statements",
    "objective":    "Sharp 2-sentence career objective statement"
  }
}

Rules:
- improvements:    6-8 items, mix of all severities
- strengths:       at least 5 items
- weaknesses:      at least 5 items
- keywordsFound:   keywords actually present in the resume
- keywordsMissing: 8-10 critical industry keywords absent from the resume
- flowSteps:       6-8 ordered steps forming a logical improvement sequence
- sampleContent:   ALL fields must be populated with rich, ready-to-use content
"""


# ─── ROUTES ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve static assets (CSS, JS)."""
    return send_from_directory(".", filename)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    POST /api/analyze
    Accepts multipart/form-data with a 'resume' file field.
    Returns JSON with full analysis from Claude.
    """
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded. Send a 'resume' field."}), 400

    file = request.files["resume"]

    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    file_bytes = file.read()

    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return jsonify({"error": f"File too large. Max {MAX_FILE_SIZE_MB} MB."}), 413

    # ── Extract text ──
    resume_text = extract_text_from_file(file_bytes, file.filename)
    has_text    = bool(resume_text and len(resume_text.strip()) > 80)

    print(f"[INFO] File: {file.filename} | Extracted: {len(resume_text)} chars | Has text: {has_text}")

    # ── Build prompt ──
    if has_text:
        user_msg = (
            f"Analyze this resume. Return ONLY raw JSON:\n\n"
            f"{resume_text[:7000]}"
        )
    else:
        user_msg = (
            f"The user uploaded a resume named '{file.filename}'. "
            f"No text could be extracted (possibly a scanned PDF). "
            f"Generate a realistic sample analysis. Return ONLY raw JSON."
        )

    # ── Call Claude ──
    try:
        response = client.messages.create(
            model      = CLAUDE_MODEL,
            max_tokens = 2000,
            system     = ANALYSIS_SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_msg}]
        )

        raw_text = "".join(
            block.text for block in response.content
            if hasattr(block, "text")
        )

        data = parse_json_response(raw_text)
        data["_meta"] = {
            "filename":   file.filename,
            "chars":      len(resume_text),
            "has_text":   has_text,
            "model":      CLAUDE_MODEL,
            "input_tokens":  response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return jsonify(data)

    except anthropic.APIError as e:
        print(f"[ERR] Anthropic API error: {e}")
        return jsonify({"error": f"AI API error: {str(e)}"}), 502

    except (ValueError, json.JSONDecodeError) as e:
        print(f"[ERR] JSON parse error: {e}")
        return jsonify({"error": "Failed to parse AI response as JSON."}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/api/regen", methods=["POST"])
def regen_section():
    """
    POST /api/regen
    Body JSON: { "section": "summary"|"experience"|"skills"|"achievements"|"objective",
                 "resume_text": "...",
                 "detected_role": "..." }
    Returns: { "content": "..." }
    """
    body         = request.get_json(silent=True) or {}
    section      = body.get("section", "summary")
    resume_text  = body.get("resume_text", "")
    detected_role = body.get("detected_role", "professional")

    context = (
        f"Based on this resume:\n\n{resume_text[:4000]}"
        if resume_text and len(resume_text.strip()) > 80
        else f"For a {detected_role} resume."
    )

    prompts = {
        "summary": (
            f"{context}\n\n"
            "Write a compelling 3-sentence professional summary. "
            "Be specific, results-oriented, and impactful. "
            "Return ONLY the text, no labels or headings."
        ),
        "experience": (
            f"{context}\n\n"
            "Write 3 powerful bullet points for their most recent role using the XYZ formula "
            "(Accomplished X by doing Y which resulted in Z). Include specific metrics where possible. "
            "Return ONLY the bullet points, starting each with •."
        ),
        "skills": (
            f"{context}\n\n"
            "List 12 relevant technical and soft skills in 1-2 natural sentences. "
            "Include both hard and soft skills. Return ONLY the text."
        ),
        "achievements": (
            f"{context}\n\n"
            "Write 2 specific, quantified achievement statements that would impress a recruiter. "
            "Use numbers, percentages, or concrete outcomes. Return ONLY the text."
        ),
        "objective": (
            f"{context}\n\n"
            "Write a sharp 2-sentence career objective statement. "
            "Be specific about the role and value offered. Return ONLY the text."
        ),
    }

    prompt = prompts.get(section, f"{context}\n\nWrite a {section} section for this resume. Return ONLY the text.")

    try:
        response = client.messages.create(
            model      = CLAUDE_MODEL,
            max_tokens = 500,
            messages   = [{"role": "user", "content": prompt}]
        )
        content = "".join(
            block.text for block in response.content
            if hasattr(block, "text")
        ).strip()
        return jsonify({"content": content})

    except anthropic.APIError as e:
        return jsonify({"error": f"AI API error: {str(e)}"}), 502

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return jsonify({
        "status":       "ok",
        "model":        CLAUDE_MODEL,
        "pdf_support":  PDF_SUPPORT,
        "docx_support": DOCX_SUPPORT,
    })


# ─── ENTRY POINT ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "true").lower() == "true"

    print("╔══════════════════════════════════════════╗")
    print("║  RESUMAI Flask Backend                   ║")
    print(f"║  Running on http://localhost:{port}         ║")
    print(f"║  Model: {CLAUDE_MODEL:<32}║")
    print(f"║  PDF support:  {'YES' if PDF_SUPPORT else 'NO (pip install pdfminer.six)'}                   ║")
    print(f"║  DOCX support: {'YES' if DOCX_SUPPORT else 'NO (pip install python-docx)'}                   ║")
    print("╚══════════════════════════════════════════╝")

    app.run(host="0.0.0.0", port=port, debug=debug)
