# app.py - True MVP: Voice Router + Simple Endpoints
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from typing import Optional, Literal
import sqlite3
from openai import OpenAI
import base64
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import json

from dotenv import load_dotenv
import os


from pydantic import BaseModel, Field
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
    cursor = conn.cursor()

    # Create users' 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            age INTEGER,
            sex TEXT,
            height_cm INTEGER, 
            weight_kg INTEGER,
            goal TEXT -- 'lose_weight', 'gain_muscle', or 'maintain'
        )
    """)

    # log user's food entries
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_logs (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            timestamp TEXT,
            type TEXT,
            description TEXT,
            calories INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Add a new table for exercise
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exercise_logs (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            exercise_type TEXT,
            start_time TEXT,
            end_time TEXT,
            duration_seconds INTEGER,
            calories_burned INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    conn.commit()
    conn.close()

init_db()


# Define a Pydantic model for the incoming data
class VoiceCommandRequest(BaseModel):
    text: str

class User(BaseModel):
    username: str
    age: int
    sex: str
    height_cm: int
    weight_kg: int 
    goal: str

class LoginRequest(BaseModel):
    username: str

class ExerciseRequest(BaseModel):
    action: Literal['start', 'stop']
    session_id: int | None = None # session_id is only needed for the 'stop' action
    exercise_type: str = 'running' # Add this with a default value for now


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
            Examples: "going for a run", "starting workout", "exercise time", "start my run", "begin workout"
             - "stop_exercise": User wants to end their current workout  
            Examples: "stop my run", "end workout", "I'm done exercising" 
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
# Food
# =============================================================================

@app.post("/log_food_direct")
async def log_food_direct(image: UploadFile, x_username: str = Header(...)):
    """Analyze food, immediately save to DB for a specific user."""
    conn = sqlite3.connect("fitness.db")
    cursor = conn.cursor()
    
    # --- ADDED: Get user_id from username ---
    cursor.execute("SELECT id FROM users WHERE username = ?", (x_username,))
    user_record = cursor.fetchone()
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found.")
    user_id = user_record[0]
    # --- END ADDITION ---
    conn.close()
    
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
    log_food(user_id, type_val, description, calories)
    
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


def log_food(user_id: int, type_val: str, description: str, calories: int):
    """Logs food to the database FOR A SPECIFIC USER."""
    conn = sqlite3.connect("fitness.db")
    # UPDATED to include user_id in the INSERT statement
    conn.execute(
        "INSERT INTO user_logs (user_id, timestamp, type, description, calories) VALUES (?, ?, ?, ?, ?)",
        (user_id, datetime.now().isoformat(), type_val, description, int(calories))
    )
    conn.commit()
    conn.close()
    return {"status": "logged"}


@app.post("/log_previous")
async def log_previous(data: dict, x_username: str = Header(...)):
    """Directly log pre-analyzed food for a specific user."""
    try:
        conn = sqlite3.connect("fitness.db")
        cursor = conn.cursor()
        
        # First, get the user's ID from their username
        cursor.execute("SELECT id FROM users WHERE username = ?", (x_username,))
        user_record = cursor.fetchone()
        if not user_record:
            raise HTTPException(status_code=404, detail="User not found.")
        user_id = user_record[0]

        # Now, insert the log with the user_id
        cursor.execute(
            "INSERT INTO user_logs (user_id, timestamp, type, description, calories) VALUES (?, ?, ?, ?, ?)",
            (user_id, datetime.now().isoformat(), data["type"], data["description"], int(data["calories"]))
        )
        conn.commit()
        conn.close()
        return {"status": "logged", "description": data["description"], "calories": data["calories"]}
    except Exception as e:
        return {"error": str(e)}    

# =============================================================================
# Exercise
# =============================================================================

@app.post("/exercise")
async def handle_exercise(req: ExerciseRequest, x_username: str = Header(...)):
    """Starts or stops an exercise session for a user."""
    conn = sqlite3.connect("fitness.db")
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    user = get_user_by_username(x_username, conn) # Reuse our helper

    if req.action == 'start':
        # Create a new exercise log entry
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO exercise_logs (user_id, start_time, exercise_type) VALUES (?, ?, ?)",
            (user['id'], datetime.now().isoformat(), req.exercise_type)
        )
        conn.commit()
        new_session_id = cursor.lastrowid
        conn.close()
        return {"status": "exercise_started", "session_id": new_session_id}

    if req.action == 'stop':
        if not req.session_id:
            raise HTTPException(status_code=400, detail="session_id is required to stop an exercise.")
        
        # Get the start time from the DB
        cursor = conn.cursor()
        cursor.execute(
            "SELECT start_time FROM exercise_logs WHERE id = ? AND user_id = ?",
            (req.session_id, user['id'])
        )
        session = cursor.fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Active exercise session not found.")
        
        start_time = datetime.fromisoformat(session['start_time'])
        end_time = datetime.now()
        duration = end_time - start_time
        duration_seconds = int(duration.total_seconds())

        # Estimate calories burned
        # Formula: METs * user_weight_kg * duration_in_hours
        # METs for running is ~9.8
        met_value = 9.8
        duration_hours = duration_seconds / 3600.0
        calories_burned = int(met_value * user['weight_kg'] * duration_hours)

        # Update the record
        cursor.execute("""
            UPDATE exercise_logs 
            SET end_time = ?, duration_seconds = ?, calories_burned = ?
            WHERE id = ?
        """, (end_time.isoformat(), duration_seconds, calories_burned, req.session_id))
        conn.commit()
        conn.close()

        return {
            "status": "exercise_stopped",
            "duration_seconds": duration_seconds,
            "calories_burned": calories_burned
        }
    
# =============================================================================
# Stats
# =============================================================================


@app.get("/summary")
async def daily_summary(x_username: str = Header(...)):
    """Provides a personalized daily summary based on user goals."""
    conn = sqlite3.connect("fitness.db")
    try:
        user = get_user_by_username(x_username, conn)
        
        # Calculate user's target calories using our new helper
        target = calculate_target_calories(
            sex=user["sex"], age=user["age"], 
            height_cm=user["height_cm"], weight_kg=user["weight_kg"], 
            goal=user["goal"]
        )
        
        # Get calories consumed today (same logic as before)
        today = datetime.now().date().isoformat()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT SUM(calories) FROM user_logs WHERE user_id = ? AND DATE(timestamp) = ?", 
            (user["id"], today)
        )
        consumed = cursor.fetchone()[0] or 0
        
        # Calculate remaining calories
        remaining = target - consumed
        
    finally:
        conn.close()
        
    return {
        "username": x_username,
        "consumed_today": consumed,
        "target_calories": target,
        "remaining_calories": remaining,
        "goal": user["goal"]
    }


# =============================================================================
# Auth end points (User Register / Login)
# =============================================================================

@app.post("/register")
async def register_user(user: User):
    """Registers a new user."""
    conn = sqlite3.connect("fitness.db")
    cursor = conn.cursor()
    try:
        username = user.username.lower().strip()  # Convert to lowercase and trim spaces
        cursor.execute(
            "INSERT INTO users (username, age, sex, height_cm, weight_kg, goal) VALUES (?, ?, ?, ?, ?, ?)",
            (username, user.age, user.sex, user.height_cm, user.weight_kg, user.goal)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists.")
    finally:
        conn.close()
    return {"status": "User registered successfully", "username": username}

@app.post("/login")
async def login_user(req: LoginRequest):
    """Logs in a user by checking if they exist."""
    conn = sqlite3.connect("fitness.db")
    cursor = conn.cursor()
    username = req.username.lower().strip() # Convert to lowercase
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return {"status": "Login successful", "username": username}
    else:
        raise HTTPException(status_code=404, detail="User not found.")

def get_user_by_username(username: str, db: sqlite3.Connection):
    """Fetches user record from the database by username."""
    cursor = db.cursor()
    username = username.lower().strip()
    # Fetch all the fields we need now
    cursor.execute("SELECT id, age, sex, height_cm, weight_kg, goal FROM users WHERE username = ?", (username,))
    user_record = cursor.fetchone()
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found.")
    return {
        "id": user_record[0], "age": user_record[1], "sex": user_record[2],
        "height_cm": user_record[3], "weight_kg": user_record[4], "goal": user_record[5]
    }

def calculate_target_calories(sex: str, age: int, height_cm: int, weight_kg: int, goal: str) -> int:
    """
    Calculates the target daily calories based on user data and goals.
    Uses Mifflin-St Jeor for BMR and a standard activity multiplier.
    """
    # 1. Calculate BMR using Mifflin-St Jeor formula
    if sex.lower() == 'male':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:  # Assume female or other, as the formula is slightly lower
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    
    # 2. Estimate TDEE (Total Daily Energy Expenditure)
    # We'll use a simple 1.4 multiplier for light-to-moderate activity for this MVP
    activity_multiplier = 1.4
    tdee = bmr * activity_multiplier

    # 3. Adjust TDEE based on the user's goal
    target_calories = tdee
    if goal == 'lose_weight':
        target_calories -= 500  # Standard deficit for ~1 lb/week loss
    elif goal == 'gain_muscle':
        target_calories += 300  # Standard surplus for lean muscle gain

    return int(target_calories)

@app.get("/profile")
async def get_profile(x_username: str = Header(...)):
    conn = sqlite3.connect("fitness.db")
    try:
        user = get_user_by_username(x_username, conn)
        return {
            "username": x_username,
            "age": user["age"],
            "sex": user["sex"], 
            "height_cm": user["height_cm"],
            "weight_kg": user["weight_kg"],
            "goal": user["goal"]
        }
    finally:
        conn.close()

# Run with: uvicorn app:app --reload

if __name__ == "__app__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)