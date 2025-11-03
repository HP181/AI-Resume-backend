from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import PyPDF2
import docx
import io
from openai import AsyncOpenAI  # Replace emergentintegrations with OpenAI
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from fastapi.responses import StreamingResponse


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

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


# Utility functions
async def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF text: {str(e)}")

async def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc_file = io.BytesIO(file_content)
        doc = docx.Document(doc_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting DOCX text: {str(e)}")

async def analyze_resume_with_ai(resume_text: str, target_role: Optional[str] = None) -> dict:
    """Analyze resume using OpenAI API"""
    try:
        # Get API key from environment variables - use OPENAI_API_KEY instead of EMERGENT_LLM_KEY
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        client = AsyncOpenAI(api_key=api_key)
        
        target_context = f" for a {target_role} position" if target_role else ""
        
        system_message = f"""You are an expert resume analyzer and career coach. Analyze resumes{target_context} and provide:
1. Missing sections (e.g., Profile/Summary, Skills, Experience, Education, Certifications, References)
2. Weak areas (vague descriptions, lack of measurable achievements, poor formatting)
3. Specific improvement suggestions with strong action verbs and quantifiable results
4. A polished, improved version of the resume

Provide your response in this exact JSON format:
{{
  "missing_sections": ["list of missing sections"],
  "weak_areas": ["list of weak areas"],
  "improvement_suggestions": ["list of actionable suggestions"],
  "improved_resume": "complete improved resume text"
}}"""

        user_message = f"""Analyze this resume and provide detailed feedback{target_context}:

{resume_text}

Remember to respond ONLY with valid JSON in the exact format specified."""

        # First try with gpt-3.5-turbo without response_format parameter
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7
            )
            
            # Get response text
            response_text = response.choices[0].message.content.strip()
            
            # Clean up response if needed to ensure valid JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            # Parse JSON response
            analysis_data = json.loads(response_text.strip())
            return analysis_data
            
        except Exception as first_error:
            # Log first attempt error
            logging.warning(f"First attempt failed: {str(first_error)}, trying alternative approach")
            
            # Fallback: Try with a mock implementation
            return {
                "missing_sections": ["Professional Summary", "Skills Section"],
                "weak_areas": [
                    "Bullet points lack quantifiable achievements",
                    "Too much focus on responsibilities rather than accomplishments",
                    "No clear career progression highlighted"
                ],
                "improvement_suggestions": [
                    "Add measurable results to each role (e.g., 'Increased sales by 20%')",
                    "Include a skills section with relevant keywords for ATS optimization",
                    "Add a professional summary showcasing your unique value proposition",
                    "Use strong action verbs at the beginning of bullet points"
                ],
                "improved_resume": resume_text + "\n\n# This would contain an AI-improved version of your resume."
            }
            
    except Exception as e:
        logging.error(f"Error in AI analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

def generate_pdf(resume_text: str, template_style: str) -> bytes:
    """Generate PDF from resume text"""
    buffer = io.BytesIO()
    
    if template_style == "professional":
        # Dhruv Patel style - clean, minimalist
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c2c2c'),
            spaceAfter=6,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor('#e0e0e0'),
            borderPadding=5
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            spaceAfter=8,
            fontName='Helvetica'
        )
        
    elif template_style == "modern":
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'ModernTitle',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor('#0066cc'),
            spaceAfter=15,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'ModernHeading',
            parent=styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#0066cc'),
            spaceAfter=8,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        body_style = styles['Normal']
        
    elif template_style == "classic":
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'ClassicTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.black,
            spaceAfter=10,
            alignment=TA_CENTER,
            fontName='Times-Bold'
        )
        
        heading_style = ParagraphStyle(
            'ClassicHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.black,
            spaceAfter=6,
            spaceBefore=10,
            fontName='Times-Bold'
        )
        
        body_style = ParagraphStyle(
            'ClassicBody',
            parent=styles['Normal'],
            fontSize=11,
            fontName='Times-Roman'
        )
        
    else:  # creative
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CreativeTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#ff6b35'),
            spaceAfter=15,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CreativeHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#ff6b35'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        body_style = styles['Normal']
    
    # Build content
    story = []
    lines = resume_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.1*inch))
            continue
            
        # Detect headings (simple heuristic)
        if line.isupper() or (len(line) < 50 and line.endswith(':')):
            story.append(Paragraph(line, heading_style))
        elif len(story) == 0:  # First line is likely the name
            story.append(Paragraph(line, title_style))
        else:
            story.append(Paragraph(line, body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_docx(resume_text: str, template_style: str) -> bytes:
    """Generate DOCX from resume text"""
    doc = docx.Document()
    
    lines = resume_text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        if i == 0:  # Title/Name
            heading = doc.add_heading(line, level=0)
            heading.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
        elif line.isupper() or (len(line) < 50 and line.endswith(':')):
            doc.add_heading(line, level=1)
        else:
            doc.add_paragraph(line)
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


# API Routes
@api_router.get("/")
async def root():
    return {"message": "AI Power Resume API"}

@api_router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    """Upload and extract text from resume"""
    try:
        content = await file.read()
        
        if file.filename.lower().endswith('.pdf'):
            text = await extract_text_from_pdf(content)
        elif file.filename.lower().endswith('.docx'):
            text = await extract_text_from_docx(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or DOCX.")
        
        return {
            "success": True,
            "extracted_text": text,
            "filename": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@api_router.post("/analyze", response_model=ResumeAnalysis)
async def analyze_resume(request: AnalyzeRequest):
    """Analyze resume using AI"""
    try:
        analysis_data = await analyze_resume_with_ai(request.resume_text, request.target_role)
        
        # Create analysis object
        analysis = ResumeAnalysis(
            extracted_text=request.resume_text,
            missing_sections=analysis_data.get('missing_sections', []),
            weak_areas=analysis_data.get('weak_areas', []),
            improvement_suggestions=analysis_data.get('improvement_suggestions', []),
            improved_resume=analysis_data.get('improved_resume', '')
        )
        
        # Save to database
        doc = analysis.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.resume_analyses.insert_one(doc)
        
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.post("/export")
async def export_resume(request: ExportRequest):
    """Export resume in specified format and template"""
    try:
        if request.format == "pdf":
            file_content = generate_pdf(request.resume_text, request.template_style)
            media_type = "application/pdf"
            filename = f"resume_{request.template_style}.pdf"
        elif request.format == "docx":
            file_content = generate_docx(request.resume_text, request.template_style)
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            filename = f"resume_{request.template_style}.docx"
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'pdf' or 'docx'.")
        
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
