
import os
import json
import base64
import re
import time
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai


app = FastAPI(title="AI Safety Inspector API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyseRequest(BaseModel):
    image_data_url: str
    language: str = "English + Hindi"
    site: str = ""
    department: str = ""
    location: str = ""
    contractor: str = ""


class Finding(BaseModel):
    id: int
    title_en: str
    title_hi: str
    category: str
    risk: str
    observation_en: str
    observation_hi: str
    consequence_en: str
    consequence_hi: str
    precautions_en: List[str]
    precautions_hi: List[str]
    bbox: List[float] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0, le=1)
    verification_needed: bool = True


class AnalyseResponse(BaseModel):
    overall_risk: str
    summary_en: str
    summary_hi: str
    findings: List[Finding]
    limitations_en: List[str]
    limitations_hi: List[str]


SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "overall_risk": {
            "type": "STRING",
            "enum": [
                "High",
                "Medium",
                "Low",
                "No clear visible hazard",
                "Unable to assess"
            ]
        },
        "summary_en": {"type": "STRING"},
        "summary_hi": {"type": "STRING"},
        "findings": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "title_en": {"type": "STRING"},
                    "title_hi": {"type": "STRING"},
                    "category": {"type": "STRING"},
                    "risk": {
                        "type": "STRING",
                        "enum": [
                            "High",
                            "Medium",
                            "Low",
                            "Needs verification"
                        ]
                    },
                    "observation_en": {"type": "STRING"},
                    "observation_hi": {"type": "STRING"},
                    "consequence_en": {"type": "STRING"},
                    "consequence_hi": {"type": "STRING"},
                    "precautions_en": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "precautions_hi": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "bbox": {
                        "type": "ARRAY",
                        "items": {"type": "NUMBER"}
                    },
                    "confidence": {"type": "NUMBER"},
                    "verification_needed": {"type": "BOOLEAN"}
                },
                "required": [
                    "id",
                    "title_en",
                    "title_hi",
                    "category",
                    "risk",
                    "observation_en",
                    "observation_hi",
                    "consequence_en",
                    "consequence_hi",
                    "precautions_en",
                    "precautions_hi",
                    "bbox",
                    "confidence",
                    "verification_needed"
                ]
            }
        },
        "limitations_en": {
            "type": "ARRAY",
            "items": {"type": "STRING"}
        },
        "limitations_hi": {
            "type": "ARRAY",
            "items": {"type": "STRING"}
        }
    },
    "required": [
        "overall_risk",
        "summary_en",
        "summary_hi",
        "findings",
        "limitations_en",
        "limitations_hi"
    ]
}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AI Safety Inspector",
        "ai": "Gemini",
        "gemini_key_configured": bool(
            os.getenv("GEMINI_API_KEY")
        )
    }


@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Backend environment variable GEMINI_API_KEY is missing."
        )

    if not req.image_data_url.startswith("data:image/"):
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid image data URL."
        )

    try:
        header, encoded = req.image_data_url.split(",", 1)

        mime_match = re.match(
            r"data:(image/[a-zA-Z0-9.+-]+);base64$",
            header
        )

        if not mime_match:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format."
            )

        mime_type = mime_match.group(1)

        image_bytes = base64.b64decode(
            encoded,
            validate=True
        )

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image is too large. Maximum size is 10 MB."
            )

        client = genai.Client(api_key=api_key)

        prompt = f"""
You are a professional occupational health and safety visual
screening assistant.

Inspect only hazards clearly visible in this workplace photograph.

Site: {req.site}
Department: {req.department}
Location: {req.location}
Contractor: {req.contractor}

Identify visible unsafe conditions and PPE issues only when
supported by the photograph.

Do not assume that permits, LOTOTO, isolation, gas testing,
competency or inspection records are missing unless visible evidence
supports that conclusion. Mention these non-visual checks in limitations.

Never invent hazards.

For each finding, provide a tight bounding box in normalized
[x, y, width, height] coordinates from 0 to 1.

If the hazard cannot be located reliably, use [0, 0, 0, 0].

Provide practical corrective actions and potential consequences.

Risk levels are preliminary screening only, not a formal risk assessment.

Return clear English and Hindi findings.

Language: {req.language}

Return structured JSON only.
"""

        # Gemini request with one automatic retry for temporary 503 errors
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=os.getenv(
                        "GEMINI_MODEL",
                        "gemini-3.8-flash"
                    ),
                    contents=[
                        prompt,
                        genai.types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type
                        )
                    ],
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=SCHEMA,
                        temperature=0.2,
                        max_output_tokens=8192
                    )
                )

                break

            except Exception as e:
                error_text = str(e).lower()

                if attempt == 0 and (
                    "503" in error_text or
                    "unavailable" in error_text
                ):
                    print(
                        "Gemini temporarily unavailable. "
                        "Retrying in 2 seconds..."
                    )
                    time.sleep(2)
                    continue

                raise

        if not response.text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        data = json.loads(response.text)

        for i, finding in enumerate(
            data.get("findings", []),
            start=1
        ):
            finding["id"] = i

            bbox = finding.get(
                "bbox",
                [0, 0, 0, 0]
            )

            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(
                    not isinstance(v, (int, float))
                    for v in bbox
                )
            ):
                finding["bbox"] = [0, 0, 0, 0]

            else:
                x, y, w, h = [
                    max(0, min(1, float(v)))
                    for v in bbox
                ]

                finding["bbox"] = [
                    x,
                    y,
                    min(w, 1 - x),
                    min(h, 1 - y)
                ]

            finding["confidence"] = max(
                0,
                min(
                    1,
                    float(finding.get("confidence", 0))
                )
            )

        return data

    except HTTPException:
        raise

    except Exception as e:
        print("Gemini analysis error:", str(e))

        error_text = str(e).lower()

        if (
            "429" in error_text
            or "resource_exhausted" in error_text
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini usage limit reached. "
                    "Please check your API quota."
                )
            )

        if (
            "503" in error_text
            or "unavailable" in error_text
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "AI service is temporarily busy. "
                    "Please try again in a minute."
                )
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini analysis failed. "
                "Please check backend logs."
            )
        )
        
