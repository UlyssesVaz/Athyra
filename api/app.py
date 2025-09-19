# app.py - True MVP: Voice Router + Simple Endpoints
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, BackgroundTasks
from typing import Optional, Literal
import sqlite3
from openai import OpenAI
import base64
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
            -- ADD THE FOLLOWING THREE LINES --
            protein INTEGER DEFAULT 0,
            carbs INTEGER DEFAULT 0,
            fats INTEGER DEFAULT 0,
            -- END OF ADDITIONS --
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
            Examples: "going for a run", "starting workout", "exercise time", "start my run", "begin workout", "track my run"
             - "stop_exercise": User wants to end their current workout  
            Examples: "stop my run", "end workout", "I'm done exercising", "finish time"
            - "get_summary": Summary requests
            Examples: "how am I doing", "daily summary", "my calories", "show my progress"
            - "clarify": Ambiguous commands that need clarification depending on context 
            - "unknown": Everything else

            Key Rules:
            1. Always assume "this" refers to the current food being viewed unless explicitly stated as "it/that".
            2. "log it/save it/log that" = log_previous (refers to something already analyzed)
            3. "log this/save this/track this" = log_food (refers to current view)
            4. Any mention of physical activity like running, workouts, or exercising should be classified as "start_exercise" or "stop_exercise".
            5. Return confidence: "high" (>90% sure), "medium" (70-90%), "low" (<70%)

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
def update_macros_in_background(log_id: int, description: str, calories: int):
    """Fetches macros from AI and updates the DB record."""

    #currently just a mock function - replace with AI call or condense in /log_food with fine tuned model
    print(f"BACKGROUND TASK: Estimating macros for log_id {log_id}")
    macros = estimate_macros_from_food(description, calories)
    
    conn = sqlite3.connect("fitness.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE user_logs SET protein = ?, carbs = ?, fats = ? WHERE id = ?",
        (macros.get("protein", 1), macros.get("carbs", 1), macros.get("fats", 1), log_id)
    )
    conn.commit()
    conn.close()
    print(f"BACKGROUND TASK: Macros updated for log_id {log_id}")



@app.post("/log_food_direct")
async def log_food_direct(image: UploadFile, background_tasks: BackgroundTasks, x_username: str = Header(...)):
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
    log_id = log_food(user_id, type_val, description, calories)

    background_tasks.add_task(update_macros_in_background, log_id, description, calories)
    
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


def log_food(user_id: int, type_val: str, description: str, calories: int) -> int:
    """Logs food to the database and returns the new log's ID."""
    conn = sqlite3.connect("fitness.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO user_logs (user_id, timestamp, type, description, calories) VALUES (?, ?, ?, ?, ?)",
        (user_id, datetime.now().isoformat(), type_val, description, int(calories))
    )
    log_id = cursor.lastrowid  # Get the ID of the new row
    conn.commit()
    conn.close()
    return log_id


@app.post("/log_previous")
async def log_previous(data: dict, background_tasks: BackgroundTasks, x_username: str = Header(...)):
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

        description = data["description"]
        calories = int(data["calories"])

        # Now, insert the log with the user_id
        cursor.execute(
            "INSERT INTO user_logs (user_id, timestamp, type, description, calories) VALUES (?, ?, ?, ?, ?)",
            (user_id, datetime.now().isoformat(), data["type"], data["description"], int(data["calories"]))
        )
        log_id = cursor.lastrowid
        conn.commit()
        conn.close()

        background_tasks.add_task(update_macros_in_background, log_id, description, calories)
        
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

@app.get("/macro_summary")
async def get_macro_summary(x_username: str = Header(...)):
    """Get macro nutrient breakdown from food logs."""
    conn = sqlite3.connect("fitness.db")
    try:
        user = get_user_by_username(x_username, conn)
        
        # Get today's food logs
        today = datetime.now().date().isoformat()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT description, calories FROM user_logs WHERE user_id = ? AND DATE(timestamp) = ?", 
            (user["id"], today)
        )
        
        food_logs = cursor.fetchall()
        
        # Calculate macros using AI estimation
        total_protein = 0
        total_carbs = 0
        total_fats = 0
        
        for description, calories in food_logs:
            # Use AI to estimate macros from food description
            macros = estimate_macros_from_food(description, calories)
            total_protein += macros["protein"]
            total_carbs += macros["carbs"]
            total_fats += macros["fats"]
        
        # Calculate targets based on user's calorie goal
        target_calories = calculate_target_calories(
            sex=user["sex"], age=user["age"], 
            height_cm=user["height_cm"], weight_kg=user["weight_kg"], 
            goal=user["goal"]
        )
        
        # Standard macro ratios: 30% protein, 40% carbs, 30% fats
        target_protein = (target_calories * 0.30) / 4  # 4 calories per gram protein
        target_carbs = (target_calories * 0.40) / 4    # 4 calories per gram carbs
        target_fats = (target_calories * 0.30) / 9     # 9 calories per gram fat
        
        return {
            "protein": {
                "current": round(total_protein),
                "target": round(target_protein),
                "percentage": min(round((total_protein / target_protein) * 100), 100)
            },
            "carbs": {
                "current": round(total_carbs),
                "target": round(target_carbs),
                "percentage": min(round((total_carbs / target_carbs) * 100), 100)
            },
            "fats": {
                "current": round(total_fats),
                "target": round(target_fats),
                "percentage": min(round((total_fats / target_fats) * 100), 100)
            }
        }
        
    finally:
        conn.close()

def estimate_macros_from_food(description: str, calories: int):
    """
    Use AI to estimate macro breakdown from food description.
    This replaces the mock data with AI-powered estimates.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""
                Estimate the macro nutrient breakdown for this food: "{description}" with {calories} calories.
                
                Return ONLY a JSON object with this exact format:
                {{"protein": X, "carbs": Y, "fats": Z}}
                
                Where X, Y, Z are grams (integers). Make sure the macros roughly add up to the calorie count:
                - Protein: 4 calories per gram
                - Carbs: 4 calories per gram  
                - Fats: 9 calories per gram
                """
            }],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "protein": result.get("protein", 0),
            "carbs": result.get("carbs", 0),
            "fats": result.get("fats", 0)
        }
        
    except Exception as e:
        print(f"Error estimating macros: {e}")
        # Fallback to simple estimation if AI fails
        return {
            "protein": calories * 0.25 / 4,  # 25% from protein
            "carbs": calories * 0.50 / 4,    # 50% from carbs  
            "fats": calories * 0.25 / 9      # 25% from fats
        }

@app.get("/exercise_summary")
async def get_exercise_summary(x_username: str = Header(...)):
    """Get today's exercise summary from exercise logs."""
    conn = sqlite3.connect("fitness.db")
    conn.row_factory = sqlite3.Row
    try:
        user = get_user_by_username(x_username, conn)
        
        # Get today's completed exercises
        today = datetime.now().date().isoformat()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT exercise_type, duration_seconds, calories_burned, start_time, end_time
            FROM exercise_logs 
            WHERE user_id = ? AND DATE(start_time) = ? AND end_time IS NOT NULL
            ORDER BY start_time DESC
        """, (user["id"], today))
        
        exercises = cursor.fetchall()
        
        exercise_list = []
        total_calories = 0
        
        for exercise in exercises:
            duration_minutes = exercise["duration_seconds"] // 60
            calories = exercise["calories_burned"] or 0
            total_calories += calories
            
            # Check if this is a personal record (simple version - longest duration for this exercise type)
            cursor.execute("""
                SELECT MAX(duration_seconds) as max_duration
                FROM exercise_logs 
                WHERE user_id = ? AND exercise_type = ? AND end_time IS NOT NULL
            """, (user["id"], exercise["exercise_type"]))
            
            max_record = cursor.fetchone()
            is_pr = max_record and exercise["duration_seconds"] == max_record["max_duration"]
            
            # Map exercise types to emojis
            exercise_icons = {
                "running": "🏃",
                "walking": "🚶", 
                "cycling": "🚴",
                "swimming": "🏊",
                "strength": "💪",
                "yoga": "🧘",
                "other": "🏃"
            }
            
            exercise_list.append({
                "type": exercise["exercise_type"].title(),
                "icon": exercise_icons.get(exercise["exercise_type"], "🏃"),
                "duration": f"{duration_minutes} min",
                "calories": calories,
                "isPR": is_pr,
                "start_time": exercise["start_time"]
            })
        
        return {
            "exercises": exercise_list,
            "total_calories": total_calories
        }
        
    finally:
        conn.close()

@app.get("/streak_data")
async def get_streak_data(x_username: str = Header(...)):
    """Calculate streak data from user activity logs."""
    conn = sqlite3.connect("fitness.db")
    try:
        user = get_user_by_username(x_username, conn)
        
        # Get all days with activity (food logs or exercise) in the last 60 days
        cursor = conn.cursor()
        sixty_days_ago = (datetime.now() - timedelta(days=60)).date().isoformat()
        
        # Get days with food logs
        cursor.execute("""
            SELECT DISTINCT DATE(timestamp) as activity_date
            FROM user_logs 
            WHERE user_id = ? AND DATE(timestamp) >= ?
            
            UNION
            
            SELECT DISTINCT DATE(start_time) as activity_date  
            FROM exercise_logs
            WHERE user_id = ? AND DATE(start_time) >= ? AND end_time IS NOT NULL
            
            ORDER BY activity_date DESC
        """, (user["id"], sixty_days_ago, user["id"], sixty_days_ago))
        
        active_dates = [row[0] for row in cursor.fetchall()]
        
        # Calculate current streak
        current_streak = 0
        today = datetime.now().date()
        
        # Check if today has activity
        today_str = today.isoformat()
        if today_str in active_dates:
            current_streak = 1
            
            # Count backwards from today
            check_date = today - timedelta(days=1)
            while check_date.isoformat() in active_dates:
                current_streak += 1
                check_date -= timedelta(days=1)
        
        # Calculate longest streak (simplified - you might want to optimize this)
        longest_streak = current_streak
        temp_streak = 0
        
        for i, date_str in enumerate(active_dates):
            if i == 0:
                temp_streak = 1
                continue
                
            current_date = datetime.fromisoformat(date_str).date()
            previous_date = datetime.fromisoformat(active_dates[i-1]).date()
            
            if (previous_date - current_date).days == 1:
                temp_streak += 1
                longest_streak = max(longest_streak, temp_streak)
            else:
                temp_streak = 1
        
        # Generate calendar for current month
        now = datetime.now()
        first_day = now.replace(day=1)
        if now.month == 12:
            last_day = now.replace(year=now.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            last_day = now.replace(month=now.month + 1, day=1) - timedelta(days=1)
        
        calendar_data = []
        current_date = first_day
        completed_days = 0
        
        while current_date <= last_day:
            date_str = current_date.isoformat()
            is_completed = date_str in active_dates
            if is_completed:
                completed_days += 1
                
            calendar_data.append({
                "day": current_date.day,
                "completed": is_completed,
                "is_today": current_date.date() == today
            })
            current_date += timedelta(days=1)
        
        return {
            "current_streak": current_streak,
            "longest_streak": longest_streak,
            "month_progress": completed_days,
            "month_total": last_day.day,
            "calendar": calendar_data,
            "month_name": now.strftime("%B")
        }
        
    finally:
        conn.close()


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

app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Run with: uvicorn app:app --reload

