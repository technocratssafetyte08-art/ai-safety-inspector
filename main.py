import os, json, base64
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="AI Safety Inspector API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your deployed app domain in production.
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
    # Normalized [0..1] coordinates: x, y, width, height.
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
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "overall_risk": {"type":"string","enum":["High","Medium","Low","No clear visible hazard","Unable to assess"]},
    "summary_en":{"type":"string"},
    "summary_hi":{"type":"string"},
    "findings":{"type":"array","items":{
      "type":"object","additionalProperties":False,
      "properties":{
        "id":{"type":"integer"},
        "title_en":{"type":"string"},"title_hi":{"type":"string"},
        "category":{"type":"string"},
        "risk":{"type":"string","enum":["High","Medium","Low","Needs verification"]},
        "observation_en":{"type":"string"},"observation_hi":{"type":"string"},
        "consequence_en":{"type":"string"},"consequence_hi":{"type":"string"},
        "precautions_en":{"type":"array","items":{"type":"string"}},
        "precautions_hi":{"type":"array","items":{"type":"string"}},
        "bbox":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},
        "confidence":{"type":"number"},
        "verification_needed":{"type":"boolean"}
      },
      "required":["id","title_en","title_hi","category","risk","observation_en","observation_hi","consequence_en","consequence_hi","precautions_en","precautions_hi","bbox","confidence","verification_needed"]
    }},
    "limitations_en":{"type":"array","items":{"type":"string"}},
    "limitations_hi":{"type":"array","items":{"type":"string"}}
  },
  "required":["overall_risk","summary_en","summary_hi","findings","limitations_en","limitations_hi"]
}

@app.get("/health")
def health():
    return {"status":"ok","service":"AI Safety Inspector"}

@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "Server is missing OPENAI_API_KEY.")
    if not req.image_data_url.startswith("data:image/"):
        raise HTTPException(400, "Please provide a valid image data URL.")
    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""
You are a cautious occupational health and safety visual screening assistant.
Inspect only what is visibly supported by this workplace photograph.
Site context: {req.site}; department: {req.department}; location: {req.location}; contractor: {req.contractor}.
Return concise, practical English and Hindi findings.
Identify visible unsafe conditions and missing/incorrect PPE only when clearly observable.
Do not infer that a permit, isolation, gas test, competency, harness anchorage, or inspection record is absent unless visible evidence supports it. Put non-visual checks in limitations.
If no clear hazard is visible, return no invented findings and say so.
For every finding, bbox must tightly surround the visible issue using normalized image coordinates [x,y,width,height] from 0 to 1. If location cannot be bounded reliably, use [0,0,0,0] and explain uncertainty.
Risk is a preliminary screening category, not a formal risk assessment. Include immediate safe precautions and verification by a competent person.
Language requested: {req.language}.
"""
        response = client.responses.create(
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
            input=[{"role":"user","content":[
                {"type":"input_text","text":prompt},
                {"type":"input_image","image_url":req.image_data_url}
            ]}],
            text={"format":{"type":"json_schema","name":"safety_inspection","strict":True,"schema":SCHEMA}}
        )
        data = json.loads(response.output_text)
        # Sanitize/validate coordinates and sequential IDs.
        for i, f in enumerate(data.get("findings", []), start=1):
            f["id"] = i
            b = f.get("bbox", [0,0,0,0])
            if len(b) != 4 or any(not isinstance(v,(int,float)) for v in b):
                f["bbox"] = [0,0,0,0]
            else:
                x,y,w,h = [max(0,min(1,float(v))) for v in b]
                f["bbox"] = [x,y,min(w,1-x),min(h,1-y)]
            f["confidence"] = max(0,min(1,float(f.get("confidence",0))))
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
