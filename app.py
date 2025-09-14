# app.py - True MVP: Voice Router + Simple Endpoints
from fastapi import FastAPI, UploadFile, File
import sqlite3
from openai import OpenAI
import base64
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import json

from dotenv import load_dotenv
import os


from pydantic import BaseModel
import re

app = FastAPI()
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI"))

#CORS
# Add CORS middleware - CRITICAL for frontend to work
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple DB setup
def init_db():
    conn = sqlite3.connect("fitness.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_logs (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            type TEXT,
            description TEXT,
            calories INTEGER
        )
    """)
    conn.close()

init_db()


# Define a Pydantic model for the incoming voice command data
class VoiceCommandRequest(BaseModel):
    text: str

# =============================================================================
# VOICE COMMAND ROUTER v2 - NLP Intent Recognition
# =============================================================================

@app.post("/voice_command")
async def voice_command(audio: UploadFile = File(...)):
    """
    Accepts audio, transcribes it, and uses an LLM to determine user intent.
    """
    try:
        # Step 1: Transcribe audio to text with Whisper
        temp_audio_path = "temp_audio.webm"
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(await audio.read())
        
        with open(temp_audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
              model="whisper-1", 
              file=audio_file
            )
        
        user_text = transcription.text
        print(f"DEBUG: Transcribed text = '{user_text}'")

        # Step 2: Improved intent classification
        system_prompt = """
            You are an intent classifier for a fitness app. Analyze the user's text and return JSON with "action" and "confidence" fields.

            Actions:
            - "log_food": User wants to analyze AND save food they're currently looking at 
            Examples: "log this food", "save this meal", "track what I'm eating", "analyze and log this" "Log this", "can you log this?", "add this"
            - "analyze_food": User wants to see food info but not save yet
            Examples: "is this healthy", "can you analyze this", "tell me about this", "how does this fit within my diet", "how does this fit within my goals"
            - "log_previous": User wants to save the last analyzed item (requires prior analysis)
            Examples: "log it", "save it", "log that", "yes lets add that", "confirm", "add it to my log"
            - "start_exercise": Exercise related
            Examples: "going for a run", "starting workout", "exercise time"
            - "get_summary": Summary requests
            Examples: "how am I doing", "daily summary", "my calories", "show my progress"
            - "clarify": Ambiguous commands that need clarification depending on context 
            - "unknown": Everything else

            Key Rules:
            1. Always assume "this" refers to the current food being viewed unless explicitly stated as "it/that".
            2. "log it/save it/log that" = log_previous (refers to something already analyzed)
            3. "log this/save this/track this" = log_food (refers to current view)
            4. Return confidence: "high" (>90% sure), "medium" (70-90%), "low" (<70%)

            Response format: {"action": "...", "confidence": "high|medium|low"}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fixed typo: was "o4-mini" 
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )

        intent_data = json.loads(response.choices[0].message.content)
        action = intent_data.get("action", "unknown")
        print(f"DEBUG: Classified action = '{action}'")

        # Step 3: Simple return - let frontend handle routing
        return {
            "transcribed_text": user_text,
            "action": action,
            "message": intent_data.get("message", "")
        }

    except Exception as e:
        print(f"ERROR in voice_command: {e}")
        return {
            "action": "unknown", 
            "message": "Sorry, I couldn't process that. Please try again.",
            "transcribed_text": ""
        }
    
# =============================================================================
# Simple endpoints - minimal logic
# =============================================================================

@app.post("/log_food_direct")
async def log_food_direct(image: UploadFile):
    """Analyze food and immediately save to DB - no frontend memory needed"""
    
    # AI call (same as analyze_food)
    image_data = base64.b64encode(await image.read()).decode()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze the food item in the image. Your response MUST be a single line in the format: food_name|description|calories_as_integer. For example: Apple|A fresh red apple|95. Do not include any other text, explanations, or markdown."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }]
    )
    
    # Parse result (same parsing logic)
    result = response.choices[0].message.content.strip()
    print(f"DEBUG: AI Response = '{result}'")
    
    parts = []
    for line in result.split('\n'):
        if '|' in line and len(line.split('|')) == 3:
            parts = line.split('|')
            break

    if len(parts) != 3:
        print(f"ERROR: Could not parse AI response. Got: '{result}'")
        return {"error": f"AI format error. Got: '{result}'. Expected: 'food|description|calories'"}
    
    type_val = parts[0].strip()
    description = parts[1].strip()
    
    try:
        calories_str = re.findall(r'\d+', parts[2])[0]
        calories = int(calories_str)
    except (IndexError, ValueError):
        print(f"ERROR: Could not parse calories from '{parts[2]}'")
        return {"error": f"Could not parse calories from AI response: '{parts[2]}'"}

    # Always save to DB for this endpoint
    log_food(type_val, description, calories)
    
    return {"description": description, "calories": calories, "saved": True}


@app.post("/analyze_food")
async def analyze_food(image: UploadFile):
    """Analyze food only - for frontend memory storage"""
    
    # Same AI call and parsing logic as before...
    image_data = base64.b64encode(await image.read()).decode()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze the food item in the image. Your response MUST be a single line in the format: food_name|description|calories_as_integer. For example: Apple|A fresh red apple|95. Do not include any other text, explanations, or markdown."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }]
    )
    
    result = response.choices[0].message.content.strip()
    print(f"DEBUG: AI Response = '{result}'")
    
    parts = []
    for line in result.split('\n'):
        if '|' in line and len(line.split('|')) == 3:
            parts = line.split('|')
            break

    if len(parts) != 3:
        print(f"ERROR: Could not parse AI response. Got: '{result}'")
        return {"error": f"AI format error. Got: '{result}'. Expected: 'food|description|calories'"}
    
    type_val = parts[0].strip()
    description = parts[1].strip()
    
    try:
        calories_str = re.findall(r'\d+', parts[2])[0]
        calories = int(calories_str)
    except (IndexError, ValueError):
        print(f"ERROR: Could not parse calories from '{parts[2]}'")
        return {"error": f"Could not parse calories from AI response: '{parts[2]}'"}

    # Never save to DB for this endpoint
    return {"description": description, "calories": calories, "saved": False}


def log_food(type_val: str, description: str, calories: int):
    """Logs food to the database"""
    conn = sqlite3.connect("fitness.db")
    conn.execute(
        "INSERT INTO user_logs (timestamp, type, description, calories) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), type_val, description, int(calories))
    )
    conn.commit()
    conn.close()
    return {"status": "logged"}

@app.post("/log_previous")
async def log_previous(data: dict):
    """Directly log pre-analyzed food without re-analysis"""
    try:
        conn = sqlite3.connect("fitness.db")
        conn.execute(
            "INSERT INTO user_logs (timestamp, type, description, calories) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), data["type"], data["description"], int(data["calories"]))
        )
        conn.commit()
        conn.close()
        return {"status": "logged", "description": data["description"], "calories": data["calories"]}
    except Exception as e:
        return {"error": str(e)}
    

@app.post("/exercise")
async def start_exercise():
    """Placeholder for exercise"""
    return {"message": "Exercise started!"}

@app.get("/summary")
async def daily_summary():
    """Simple daily summary"""
    conn = sqlite3.connect("fitness.db")
    today = datetime.now().date().isoformat()
    
    food_calories = conn.execute(
        "SELECT SUM(calories) FROM user_logs WHERE type='food' AND DATE(timestamp)=?", 
        (today,)
    ).fetchone()[0] or 0
    
    conn.close()
    return {"calories_today": food_calories}

# Run with: uvicorn app:app --reload