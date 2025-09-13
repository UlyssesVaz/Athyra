# app.py - The entire MVP in ~50 lines
from fastapi import FastAPI, UploadFile
import sqlite3
import openai
import base64
from datetime import datetime

app = FastAPI()
openai.api_key = 

# Initialize database
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

# =============================================================================
# VOICE COMMAND ROUTER - Pure Classification Only
# =============================================================================

@app.post("/voice_command")
async def voice_command(text: str = "", audio: UploadFile = None):
    """
    Pure voice command classifier - NO business logic
    Returns instructions for frontend to execute
    
    Frontend flow:
    1. Call this endpoint with voice input
    2. Receive action instructions  
    3. Execute appropriate camera/sensor actions
    4. Call specific action endpoints with data
    """
    
    # Future: Add speech-to-text when audio is provided
    if audio:
        # TODO: Implement Whisper API
        # audio_data = await audio.read()
        # transcription = openai.Audio.transcribe("whisper-1", audio_data)
        # text = transcription.text
        pass
    
    # Clean and classify voice command
    command = text.lower().strip()
    
    if "log this food" in command or "save this food" in command:
        return {
            "action": "log_food",
            "message": "Point camera at food to log it",
            "frontend_flow": "take_photo_then_call_analyze_food",
            "endpoint": "/analyze_food",
            "parameters": {"save_to_db": True}
        }
    
    elif "what is this food" in command or "analyze this food" in command:
        return {
            "action": "analyze_food", 
            "message": "Point camera at food to analyze it",
            "frontend_flow": "take_photo_then_call_analyze_food",
            "endpoint": "/analyze_food",
            "parameters": {"save_to_db": False}
        }
    
    elif "start exercise" in command or "begin workout" in command:
        return {
            "action": "start_exercise",
            "message": "Starting exercise session",
            "frontend_flow": "call_exercise_endpoint",
            "endpoint": "/exercise",
            "parameters": {}
        }
    
    else:
        return {
            "action": "unknown",
            "message": "Command not recognized. Try: 'log this food', 'analyze this food', or 'start exercise'",
            "available_commands": [
                "log this food - saves food to your daily log",
                "analyze this food - shows nutrition info without saving", 
                "start exercise - begins exercise tracking"
            ]
        }

# =============================================================================
# FOOD ANALYSIS ENDPOINT - Pure Food Logic
# =============================================================================

@app.post("/analyze_food")
async def analyze_food(image: UploadFile, save_to_db: bool = True):
    """
    Pure food analysis - no voice logic mixed in
    Takes image, returns food analysis, optionally saves to DB
    """
    # Convert image to base64
    image_data = base64.b64encode(await image.read()).decode()
    
    # AI food analysis
    response = openai.chat.completions.create(
        model="o4-mini",
        messages=[{
            "role": "user", 
            "content": [
                {
                    "type": "text", 
                    "text": "Analyze this food image. Respond in format: TYPE|DESCRIPTION|CALORIES (e.g., 'food|grilled salmon with rice|450')"
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        }]
    )
    
    # Parse AI response
    result = response.choices[0].message.content.strip()
    try:
        type_val, description, calories = result.split('|')
        calories = int(calories)
    except ValueError:
        return {"error": "Could not analyze food properly", "raw_response": result}
    
    # Save to database if requested
    if save_to_db:
        conn = sqlite3.connect("fitness.db")
        conn.execute(
            "INSERT INTO user_logs (timestamp, type, description, calories) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), type_val.strip(), description.strip(), calories)
        )
        conn.commit()
        conn.close()
        
        return {
            "status": "logged",
            "food": description.strip(),
            "calories": calories,
            "message": f"Logged: {description.strip()} - {calories} calories"
        }
    else:
        return {
            "status": "analyzed",
            "food": description.strip(),
            "calories": calories,
            "message": f"Analysis: {description.strip()} - {calories} calories",
            "saved": False
        }

# =============================================================================
# EXERCISE ENDPOINT - Pure Exercise Logic  
# =============================================================================

@app.post("/exercise")
async def start_exercise():
    """
    Exercise tracking endpoint - completely separate from food logic
    """
    # Placeholder for now - will handle exercise session tracking
    return {
        "status": "started",
        "message": "Exercise tracking started!",
        "session_id": f"session_{datetime.now().timestamp()}",
        "note": "Full exercise tracking coming in Phase 3!"
    }

@app.get("/summary")
async def daily_summary():
    conn = sqlite3.connect("fitness.db")
    today = datetime.now().date().isoformat()
    
    food_calories = conn.execute(
        "SELECT SUM(calories) FROM user_logs WHERE type='food' AND DATE(timestamp)=?", 
        (today,)
    ).fetchone()[0] or 0
    
    exercise_calories = conn.execute(
        "SELECT SUM(calories) FROM user_logs WHERE type='exercise' AND DATE(timestamp)=?", 
        (today,)
    ).fetchone()[0] or 0
    
    conn.close()
    return {"food": food_calories, "exercise": exercise_calories, "net": food_calories - exercise_calories}


# Run with: uvicorn app:app --reload