from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import io
import json
from fastapi.responses import StreamingResponse

# Create the main app
app = FastAPI()

# Add CORS middleware with more explicit configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-resume-frontend-nu.vercel.app", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class ResumeAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    extracted_text: str
    missing_sections: List[str]
    weak_areas: List[str]
    improvement_suggestions: List[str]
    improved_resume: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AnalyzeRequest(BaseModel):
    resume_text: str
    target_role: Optional[str] = None

class ExportRequest(BaseModel):
    resume_text: str
    template_style: str  # "modern", "classic", "creative", "professional"
    format: str  # "pdf" or "docx"

# Helper functions for file processing
async def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        import PyPDF2
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logging.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {str(e)}")

async def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        import docx
        doc_file = io.BytesIO(file_content)
        doc = docx.Document(doc_file)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text.strip()
    except Exception as e:
        logging.error(f"DOCX extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error extracting DOCX text: {str(e)}")

async def analyze_resume_with_ai(resume_text: str, target_role: Optional[str] = None) -> dict:
    """Analyze resume using OpenAI API"""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise HTTPException(status_code=500, detail="Missing OpenAI API key")

        # Initialize the OpenAI client (no proxies argument)
        client = OpenAI(api_key=api_key)

        target_context = f" for a {target_role} position" if target_role else ""

        system_message = f"""You are an expert resume analyzer and career coach. Analyze resumes{target_context} and provide:
1. Missing sections
2. Weak areas
3. Actionable improvement suggestions
4. A polished, improved version of the resume

Respond ONLY in valid JSON:
{{
  "missing_sections": ["list"],
  "weak_areas": ["list"],
  "improvement_suggestions": ["list"],
  "improved_resume": "text"
}}"""

        user_message = f"Analyze this resume and provide detailed feedback{target_context}:\n\n{resume_text}"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
        )

        response_text = response.choices[0].message.content.strip()

        # Clean up markdown JSON formatting if present
        response_text = response_text.strip("` \n")
        if response_text.startswith("json"):
            response_text = response_text[4:]

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as err:
            logging.error(f"JSON parse error: {err}")
            raise HTTPException(status_code=500, detail="Invalid AI response format")

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in AI analysis: {str(e)}")
        raise HTTPException(status_code=503, detail=f"AI analysis service error: {str(e)}")

# Routes
@api_router.get("/")
async def root():
    return {"message": "AI Power Resume API"}

@api_router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    """Upload and extract text from resume"""
    try:
        content = await file.read()

        if file.filename.lower().endswith(".pdf"):
            text = await extract_text_from_pdf(content)
        elif file.filename.lower().endswith(".docx"):
            text = await extract_text_from_docx(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        return {"success": True, "extracted_text": text, "filename": file.filename}
    except Exception as e:
        logging.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@api_router.post("/analyze", response_model=ResumeAnalysis)
async def analyze_resume(request: AnalyzeRequest):
    """Analyze resume using AI"""
    try:
        logging.info(f"Analyzing resume{' for ' + request.target_role if request.target_role else ''}")
        analysis_data = await analyze_resume_with_ai(request.resume_text, request.target_role)

        analysis = ResumeAnalysis(
            extracted_text=request.resume_text,
            missing_sections=analysis_data.get("missing_sections", []),
            weak_areas=analysis_data.get("weak_areas", []),
            improvement_suggestions=analysis_data.get("improvement_suggestions", []),
            improved_resume=analysis_data.get("improved_resume", ""),
        )

        return analysis
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Include router
app.include_router(api_router)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
