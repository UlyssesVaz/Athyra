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
    This is the new "brain" of your app.
    """
    try:
        # Step 1: Transcribe audio to text with Whisper
        # We save the audio to a temporary file because the transcription API needs a file path.
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

        # Step 2: Classify the intent of the text using GPT-4o
        system_prompt = """
            You are an intent recognition router for a voice-controlled fitness app.
            Your task is to analyze the user's text and determine their intent.
            Your response MUST be a JSON object with a single key: "action".
            The possible values for "action" are: "log_food", "analyze_food", "start_exercise", "get_summary", or "unknown".

            Examples:
            - User: "Please log this meal for me." -> {"action": "log_food"}
            - User: "What is this food?" -> {"action": "analyze_food"}
            - User: "how many calories have i had" -> {"action": "get_summary"}
            - User: "i'm going for a run" -> {"action": "start_exercise"}
            - User: "what's the weather like" -> {"action": "unknown"}
        """

        response = client.chat.completions.create(
            model="o4-mini",
            response_format={ "type": "json_object" }, # Use JSON mode for reliable output
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )

        intent_data = json.loads(response.choices[0].message.content)
        action = intent_data.get("action", "unknown")
        print(f"DEBUG: Classified action = '{action}'")

        # Step 3: Return the action and necessary parameters to the frontend
        if action == "log_food":
            return {"action": "log_food", "save_to_db": True}
        elif action == "analyze_food":
            return {"action": "analyze_food", "save_to_db": False}
        elif action == "start_exercise":
            return {"action": "start_exercise"}
        elif action == "get_summary":
            return {"action": "get_summary"} # We'll need a frontend handler for this
        else:
            return {"action": "unknown", "message": f"I'm not sure how to handle '{user_text}'"}

    except Exception as e:
        print(f"ERROR in voice_command: {e}")
        return {"action": "unknown", "message": "An error occurred while processing your voice."}

# =============================================================================
# Simple endpoints - minimal logic
# =============================================================================

@app.post("/analyze_food")
async def analyze_food(image: UploadFile, save_to_db: bool = False):
    """Simple food analysis"""
    
    # AI call
    image_data = base64.b64encode(await image.read()).decode()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                # propmt engineering to ensure strict format. REWORK WITH MICRONUTRIENTS LATER
                {"type": "text", "text": "Analyze the food item in the image. Your response MUST be a single line in the format: food_name|description|calories_as_integer. For example: Apple|A fresh red apple|95. Do not include any other text, explanations, or markdown."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }]
    )
    
    # Parse result with error handling
    
    result = response.choices[0].message.content.strip()
    print(f"DEBUG: AI Response = '{result}'")
    



    # --- Call back / guardrails ---
    parts = []
    # Find the line that contains our data, in case the AI adds extra lines
    for line in result.split('\n'):
        if '|' in line and len(line.split('|')) == 3:
            parts = line.split('|')
            break # We found it, stop looking

    if len(parts) != 3:
        print(f"ERROR: Could not parse AI response. Got: '{result}'")
        return {"error": f"AI format error. Got: '{result}'. Expected: 'food|description|calories'"}
    
    # Sanitize the parts
    type_val = parts[0].strip()
    description = parts[1].strip()
    
    # Extract just the numbers for calories
    try:
        # Use regex to find the first number in the string
        calories_str = re.findall(r'\d+', parts[2])[0]
        calories = int(calories_str)
    except (IndexError, ValueError):
        print(f"ERROR: Could not parse calories from '{parts[2]}'")
        return {"error": f"Could not parse calories from AI response: '{parts[2]}'"}
    # --- PARSING LOGIC CHANGES END HERE ---


    
    # Maybe save
    if save_to_db:
        conn = sqlite3.connect("fitness.db")
        conn.execute(
            "INSERT INTO user_logs (timestamp, type, description, calories) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), type_val, description, int(calories))
        )
        conn.commit()
        conn.close()
    
    return {"description": description, "calories": int(calories), "saved": save_to_db}

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