import os
import re
import json
import random
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

# Load .env relative to this file's directory, with fallback to cwd
env_file = Path(__file__).resolve().parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
MODEL_ID = "gemini-2.5-flash"


def rule_based_recommendation(report, conditions):
    return (
        "### CLINICAL ASSESSMENT\n"
        f"* **Diagnostic Evaluation**: Overall barrier integrity is stable at {report.get('skin_health', 75):.1f}%. "
        f"Active blemish index is {report.get('acne', 15):.1f}%, with pore texture score at {report.get('texture', 50):.1f}%.\n"
        f"* **Inferred Classification**: {conditions.get('Skin Type', 'Combination')} skin type with {conditions.get('Hydration Level', 'Moderate')} hydration levels.\n\n"
        "### MORNING PROTOCOL\n"
        "1. **Cleanser**: Gentle, low-pH amino acid foaming wash.\n"
        "2. **Target Active**: 15% L-Ascorbic Acid (Vitamin C) + 0.5% Ferulic Acid serum.\n"
        "3. **Barrier Hydration**: Lightweight Hyaluronic Acid & Polyglutamic humectant gel.\n"
        "4. **Photoprotection**: Broad-Spectrum Mineral SPF 50+ PA++++.\n\n"
        "### EVENING REPAIR PROTOCOL\n"
        "1. **Cleanse**: Non-stripping lipid-replenishing milk cleanser.\n"
        "2. **Targeted Active**: Encapsulated Retinaldehyde 0.05% or Copper Tripeptide-1 complex.\n"
        "3. **Lipid Barrier Recovery**: Physiological Ceramide NP/AP/EOP + Squalane emulsion.\n\n"
        "### HERO ACTIVE FORMULATIONS\n"
        "* **Retinaldehyde**: Accelerates cellular turnover and dermal collagen synthesis to refine micro-texture.\n"
        "* **L-Ascorbic Acid**: Neutralizes oxidative stress and inhibits melanin transfer.\n"
        "* **Ceramides**: Restores stratum corneum lipid lamellae to enhance transepidermal water retention.\n\n"
        "### CLINICAL DIRECTIVE\n"
        "Daily broad-spectrum photoprotection is mandatory to safeguard collagen integrity and optimize barrier repair."
    )


def clean_clinical_output(raw_text):
    """Enforces strictly clinical markdown structure with proper headings."""
    if not raw_text:
        return raw_text

    # Strip conversational chatter at beginning
    cleaned = re.sub(
        r"^(As a\s+[^\n]+|Here is\s+[^\n]+|I have reviewed\s+[^\n]+|Based on your\s+[^\n]+|Hello[^\n]+|Certainly[^\n]+|Sure[^\n]+)\n*",
        "",
        raw_text.strip(),
        flags=re.IGNORECASE
    )
    # Remove leading horizontal rules
    cleaned = re.sub(r"^(\s*---\s*|\s*\*\*\*\s*)+", "", cleaned).strip()

    # Ensure main headings start with markdown ### for consistent card rendering
    section_names = [
        "CLINICAL ASSESSMENT",
        "MORNING PROTOCOL",
        "EVENING REPAIR PROTOCOL",
        "HERO ACTIVE FORMULATIONS",
        "CLINICAL DIRECTIVE"
    ]
    for sec in section_names:
        # Match "1. CLINICAL ASSESSMENT" or "### CLINICAL ASSESSMENT" or "CLINICAL ASSESSMENT"
        pattern = re.compile(rf"(?:\n|^)(?:[#\d\.\s\*\-]+)?\s*({sec})\s*(?:\n|:)", re.IGNORECASE)
        cleaned = pattern.sub(rf"\n\n### \1\n", cleaned)

    return cleaned.strip()


def get_product_recommendations(report, lifestyle, conditions):
    """
    Calls Gemini 2.5 Flash via high-performance REST API.
    Enforces strictly clinical, non-conversational medical formatting with proper markdown hierarchy.
    """
    global api_key
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return rule_based_recommendation(report, conditions)

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={api_key}"
        prompt = f"""
Act as a clinical cosmetic dermatology consultation system. Formulate a structured medical skincare protocol based on this objective DermaCLIP diagnostic.

[PATIENT DERMACLIP METRICS (0-100 scale)]
- Acne / Blemish Activity: {report.get('acne', 0):.1f} / 100
- Hyperpigmentation / Melanin Clusters: {report.get('pigmentation', 0):.1f} / 100
- Wrinkle Depth & Fine Lines: {report.get('wrinkles', 0):.1f} / 100
- Under-Eye Dark Circles & Congestion: {report.get('dark_circles', 0):.1f} / 100
- Pores & Micro-Texture Irregularity: {report.get('texture', 0):.1f} / 100
- Overall Skin Barrier Health: {report.get('skin_health', 0):.1f} / 100
- Inferred Classification: {conditions}

[MANDATORY FORMAT RULES]
1. ZERO CONVERSATIONAL PREAMBLE: Do NOT write "As a board-certified dermatologist...", "I have reviewed...", or any greeting.
2. START DIRECTLY with the heading: "### CLINICAL ASSESSMENT"
3. Use clean markdown formatting with bold parameter labels:
   * **Primary Etiology**: [analysis]
   * **Secondary Etiology**: [analysis]
   * **Inferred Classification**: [analysis]
4. For protocols, format as numbered steps:
   1. **Cleanser**: [name and formula]
   2. **Target Active**: [exact ingredient and concentration %]
   3. **Barrier Hydration**: [exact formulation]
   4. **Photoprotection**: [SPF specification]
5. For active ingredients:
   * **[Ingredient Name]**: [exact mechanism matching the patient's scores]
6. For clinical directive:
   Provide one concise, evidence-based behavioral directive.

Keep formatting impeccably structured with clear markdown headings (###).
"""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": 2500
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=18) as response:
            result = json.loads(response.read().decode("utf-8"))
            llm_text = result["candidates"][0]["content"]["parts"][0]["text"]
            return clean_clinical_output(llm_text)

    except Exception as e:
        print(f"⚠️ Gemini API fallback: {e}")
        return rule_based_recommendation(report, conditions)
