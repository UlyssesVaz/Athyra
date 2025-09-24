# app.py - True MVP: Voice Router + Simple Endpoints
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, BackgroundTasks
from typing import Optional, Literal
import sqlite3
from openai import OpenAI
import base64
import asyncio
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

    # Add a new table for meal plans
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meal_plans (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            created_at TEXT,
            plan_data TEXT,
            is_active INTEGER,
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
# Agent Classes for Meal Planning Pipeline
# =============================================================================

class MealPlanAgent:
    """Agent 1: Generates weekly meal plan based on user data"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    async def generate_plan(self, user_data):
        """Generate meal plan from user preferences and history"""
        
        # Prepare context from user's food history
        food_context = self._build_food_context(user_data.get('food_history', []))
        
        prompt = f"""
        Create a 7-day meal plan for a user with these preferences:
        - Goal: {user_data.get('goal', 'maintain')}
        - Budget: ${user_data.get('budget', 100)}/week
        - Allergies: {user_data.get('allergies', 'none')}
        - Food History: {food_context}
        - Target Calories: {user_data.get('target_calories', 2000)}/day
        
        Return JSON format:
        {{
            "week_plan": {{
                "day_1": {{
                    "breakfast": {{"recipe": "...", "calories": 0, "prep_time": "..."}},
                    "lunch": {{"recipe": "...", "calories": 0, "prep_time": "..."}},
                    "dinner": {{"recipe": "...", "calories": 0, "prep_time": "..."}}
                }},
                // ... repeat for 7 days
            }},
            "total_weekly_calories": 0,
            "reasoning": "Why these meals fit the user's profile"
        }}
        """
        
        response = await self._call_openai(prompt)
        return json.loads(response)
    
    def _build_food_context(self, food_history):
        """Summarize user's eating patterns"""
        if not food_history:
            return "No previous food history"
        
        recent_foods = food_history[-10:]  # Last 10 entries
        return f"Recent foods: {', '.join([food['description'] for food in recent_foods])}"
    
    async def _call_openai(self, prompt: str) -> str:
        """Helper method for OpenAI API calls"""
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model="o4-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

class ShoppingListAgent:
    """Agent 2: Converts meal plan to consolidated grocery list"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    async def compile_list(self, meal_plan):
        """Convert meal plan to grocery list with quantities"""
        
        prompt = f"""
        Convert this meal plan to a consolidated grocery list:
        {json.dumps(meal_plan['week_plan'])}
        
        Consolidate ingredients (e.g., if multiple recipes need eggs, calculate total needed).
        Estimate realistic quantities for grocery shopping.
        
        Return JSON format:
        {{
            "grocery_list": [
                {{"item": "chicken breast", "quantity": "2 lbs", "category": "meat"}},
                {{"item": "rice", "quantity": "5 lbs", "category": "grains"}},
                // ...
            ],
            "estimated_cost": 0,
            "shopping_categories": ["produce", "meat", "dairy", "pantry"]
        }}
        """
        
        response = await self._call_openai(prompt)
        return json.loads(response)
    
    async def _call_openai(self, prompt: str) -> str:
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model="o4-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

class HealthValidatorAgent:
    """Agent 3: Validates nutritional completeness and safety"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    async def validate_plan(self, meal_plan, shopping_list, user_data):
        """Analyze nutritional completeness and flag potential issues"""
        
        prompt = f"""
        Analyze this meal plan for nutritional completeness:
        Meal Plan: {json.dumps(meal_plan)}
        Shopping List: {json.dumps(shopping_list)}
        User Info: Goal={user_data.get('goal')}, Allergies={user_data.get('allergies')}
        
        Check for:
        1. Macro balance (protein/carbs/fats)
        2. Micronutrient coverage
        3. Allergy conflicts
        4. Calorie appropriateness
        
        Return JSON format:
        {{
            "health_score": 0-100,
            "nutritional_analysis": {{
                "protein_adequacy": "adequate|low|high",
                "micronutrient_gaps": ["vitamin_d", "iron"],
                "macro_distribution": {{"protein": "25%", "carbs": "45%", "fats": "30%"}}
            }},
            "warnings": ["Potential allergy risk with...", "Low in vitamin B12"],
            "improvements": ["Add more leafy greens", "Include fish twice per week"],
            "approval_status": "approved|needs_revision"
        }}
        """
        
        response = await self._call_openai(prompt)
        return json.loads(response)
    
    async def _call_openai(self, prompt: str) -> str:
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

class BudgetOptimizerAgent:
    """Agent 4: Optimizes shopping list within budget constraints"""
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    async def optimize_budget(self, shopping_list, health_analysis, budget):
        """Optimize shopping list for budget while maintaining nutrition"""
        
        prompt = f"""
        Optimize this shopping list within budget ${budget}:
        Shopping List: {json.dumps(shopping_list)}
        Health Analysis: {json.dumps(health_analysis)}
        
        Priority rules:
        1. Don't compromise on allergy safety
        2. Maintain protein adequacy
        3. Suggest cheaper alternatives for expensive items
        4. Remove/reduce non-essential items if over budget
        
        Return JSON format:
        {{
            "optimized_list": [
                {{"item": "chicken thighs", "quantity": "2 lbs", "price": 8.99, "substituted_from": "chicken breast"}},
                // ...
            ],
            "total_cost": 0,
            "savings": 0,
            "substitutions_made": [
                {{"original": "chicken breast", "replacement": "chicken thighs", "reason": "50% cost savings, similar protein"}}
            ],
            "removed_items": ["expensive_spice"],
            "budget_status": "under|over|exact"
        }}
        """
        
        response = await self._call_openai(prompt)
        return json.loads(response)
    
    async def _call_openai(self, prompt: str) -> str:
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

class MealPlanOrchestrator:
    """Coordinates all agents and manages the pipeline"""
    
    def __init__(self, openai_client):
        self.meal_agent = MealPlanAgent(openai_client)
        self.shopping_agent = ShoppingListAgent(openai_client)
        self.health_agent = HealthValidatorAgent(openai_client)
        self.budget_agent = BudgetOptimizerAgent(openai_client)
    
    async def create_meal_plan(self, user_data):
        """Execute full meal planning pipeline"""
        
        start_time = datetime.now()
        pipeline_results = {}
        
        try:
            # Agent 1: Generate meal plan
            print("Agent 1: Generating meal plan...")
            meal_plan = await self.meal_agent.generate_plan(user_data)
            pipeline_results['meal_plan'] = meal_plan
            
            # Agent 2: Create shopping list
            print("Agent 2: Creating shopping list...")
            shopping_list = await self.shopping_agent.compile_list(meal_plan)
            pipeline_results['shopping_list'] = shopping_list
            
            # Agent 3: Health validation
            print("Agent 3: Validating health...")
            health_analysis = await self.health_agent.validate_plan(meal_plan, shopping_list, user_data)
            pipeline_results['health_analysis'] = health_analysis
            
            # Agent 4: Budget optimization
            print("Agent 4: Optimizing budget...")
            budget_optimization = await self.budget_agent.optimize_budget(
                shopping_list, health_analysis, user_data.get('budget', 100)
            )
            pipeline_results['budget_optimization'] = budget_optimization
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            pipeline_results['execution_metrics'] = {
                'total_time_seconds': execution_time,
                'agent_calls': 4,
                'estimated_cost': execution_time * 0.002  # Rough cost estimate
            }
            
            return pipeline_results
            
        except Exception as e:
            return {
                'error': str(e),
                'partial_results': pipeline_results,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }

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

# app.py

@app.get("/macro_summary")
async def get_macro_summary(x_username: str = Header(...)):
    """Get macro nutrient breakdown from saved food log data."""
    conn = sqlite3.connect("fitness.db")
    try:
        user = get_user_by_username(x_username, conn)
        today = datetime.now().date().isoformat()
        cursor = conn.cursor()

        # Fetch the SUM of macros directly from the database
        cursor.execute("""
            SELECT SUM(protein), SUM(carbs), SUM(fats) 
            FROM user_logs 
            WHERE user_id = ? AND DATE(timestamp) = ?
        """, (user["id"], today))
        
        totals = cursor.fetchone()
        total_protein = totals[0] or 0
        total_carbs = totals[1] or 0
        total_fats = totals[2] or 0
        
        # Calculate targets based on user's calorie goal
        target_calories = calculate_target_calories(
            sex=user["sex"], age=user["age"], 
            height_cm=user["height_cm"], weight_kg=user["weight_kg"], 
            goal=user["goal"]
        )
        
        # Standard macro ratios: 30% protein, 40% carbs, 30% fats
        target_protein = (target_calories * 0.30) / 4
        target_carbs = (target_calories * 0.40) / 4
        target_fats = (target_calories * 0.30) / 9
        
        # Prevent division by zero if targets are 0
        protein_perc = (total_protein / target_protein * 100) if target_protein > 0 else 0
        carbs_perc = (total_carbs / target_carbs * 100) if target_carbs > 0 else 0
        fats_perc = (total_fats / target_fats * 100) if target_fats > 0 else 0

        return {
            "protein": {
                "current": round(total_protein),
                "target": round(target_protein),
                "percentage": min(round(protein_perc), 100)
            },
            "carbs": {
                "current": round(total_carbs),
                "target": round(target_carbs),
                "percentage": min(round(carbs_perc), 100)
            },
            "fats": {
                "current": round(total_fats),
                "target": round(target_fats),
                "percentage": min(round(fats_perc), 100)
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

# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/create_meal_plan")
async def create_meal_plan(
    req: dict, # Updated to receive a dict
    x_username: str = Header(...)
):
    """Generate and SAVE a complete meal plan using the multi-agent system."""
    
    conn = sqlite3.connect("fitness.db")
    try:
        user = get_user_by_username(x_username, conn)
        budget = req.get('budget', 100)
        allergies = req.get('allergies', "")

        # ... (The existing logic to get food_history, target_calories, etc. remains the same)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT description, calories, timestamp FROM user_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20",
            (user["id"],)
        )
        food_history = [{"description": row[0], "calories": row[1], "timestamp": row[2]} for row in cursor.fetchall()]
        target_calories = calculate_target_calories(sex=user["sex"], age=user["age"], height_cm=user["height_cm"], weight_kg=user["weight_kg"], goal=user["goal"])
        
        user_data = {
            'goal': user['goal'],
            'budget': budget,
            'allergies': allergies.split(',') if allergies else [],
            'target_calories': target_calories,
            'food_history': food_history
        }
        
        orchestrator = MealPlanOrchestrator(client)
        results = await orchestrator.create_meal_plan(user_data)
        
        # --- NEW LOGIC TO SAVE THE PLAN ---
        if 'error' not in results:
            # 1. Deactivate any old plans for this user
            cursor.execute("UPDATE meal_plans SET is_active = 0 WHERE user_id = ?", (user["id"],))
            
            # 2. Insert the new plan as a JSON string
            cursor.execute("""
                INSERT INTO meal_plans (user_id, created_at, plan_data, is_active)
                VALUES (?, ?, ?, ?)
            """, (user["id"], datetime.now().isoformat(), json.dumps(results), 1))
            
            conn.commit()        
        return results
        
    finally:
        conn.close()

@app.get("/meal_plan_status")
async def get_meal_plan_status(x_username: str = Header(...)):
    """Get current meal plan status (placeholder for future persistence)"""
    return {
        "status": "no_active_plan",
        "last_generated": None,
        "plan_id": None
    }

@app.get("/get_active_meal_plan")
async def get_active_meal_plan(x_username: str = Header(...)):
    """Fetches the current active meal plan for a user from the database."""
    conn = sqlite3.connect("fitness.db")
    try:
        user = get_user_by_username(x_username, conn)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT plan_data FROM meal_plans WHERE user_id = ? AND is_active = 1 ORDER BY created_at DESC LIMIT 1",
            (user["id"],)
        )
        active_plan = cursor.fetchone()
        
        if active_plan:
            # The data is stored as a string, so we parse it back into JSON
            return json.loads(active_plan[0])
        else:
            # No active plan found
            raise HTTPException(status_code=404, detail="No active meal plan found.")
            
    finally:
        conn.close()

@app.get("/get_all_meal_plans")
async def get_all_meal_plans(x_username: str = Header(...)):
    """Fetches all meal plans a user has ever created."""
    conn = sqlite3.connect("fitness.db")
    try:
        user = get_user_by_username(x_username, conn)
        cursor = conn.cursor()
        
        # Select the id, creation date, and plan data for all plans, newest first.
        cursor.execute(
            "SELECT id, created_at, plan_data FROM meal_plans WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],)
        )
        all_plans = cursor.fetchall()
        
        # Return a list of plans
        return [{
            "plan_id": row[0],
            "created_at": row[1],
            "plan_data": json.loads(row[2]) # Parse the JSON data before sending
        } for row in all_plans]
            
    except Exception as e:
        print(f"Error fetching all meal plans: {e}")
        return [] # Return an empty list on error
    finally:
        conn.close()

app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Run with: uvicorn app:app --reload

