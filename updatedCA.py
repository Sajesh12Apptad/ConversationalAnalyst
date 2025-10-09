from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import pandas as pd
import io
import os
from typing import List, Dict, Any, Optional
import re
import requests
import json
import hashlib, time
from difflib import SequenceMatcher
from collections import OrderedDict

# ===================== Optimization Layer =====================

CACHE = OrderedDict()
CACHE_TTL = 0  # 24 hours

TEMPLATES = {
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
    "goodbye": "Goodbye! Take care.",
    "hello": "Hello! How can I help you today?"
}

def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()

def is_similar(a: str, b: str, threshold: float = 0.9) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def get_cached_response(prompt: str):
    now = time.time()
    normalized = prompt.strip().lower()

    if normalized in TEMPLATES:
        return TEMPLATES[normalized]
    
    h = hash_prompt(prompt)
    if h in CACHE:
        if now - CACHE[h]['time'] < CACHE_TTL:
            return CACHE[h]['response']
        else:
            del CACHE[h]

    for entry in CACHE.values():
        if is_similar(prompt, entry['prompt']):
            return entry['response']
    
    return None

def save_to_cache(prompt: str, response: str):
    h = hash_prompt(prompt)
    CACHE[h] = {
        'prompt': prompt,
        'response': response,
        'time': time.time()
    }


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "app_data.db"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

class QueryRequest(BaseModel):
    prompt: str
    conversation_history: Optional[List[Dict]] = None
    selected_dataset: Optional[str] = None
    conversation_state: Optional[Dict] = None
    mode: Optional[str] = "auto"  # "auto", "analysis", "dashboard"

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

def get_available_tables():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    table_info = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info([{table}])")
        columns = [{"name": col[1], "type": col[2]} for col in cursor.fetchall()]
        table_info[table] = columns
    conn.close()
    return table_info

def get_table_sample_data(table_name: str, limit: int = 5) -> List[Dict]:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM [{table_name}] LIMIT {limit}")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    results = [dict(zip(columns, r)) for r in rows]
    conn.close()
    return results

def get_dataset_summary(table_name: str) -> Dict[str, Any]:
    """Get comprehensive dataset summary for AI context"""
    conn = get_conn()
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
    row_count = cursor.fetchone()[0]
    
    cursor.execute(f"PRAGMA table_info([{table_name}])")
    columns = [{"name": col[1], "type": col[2]} for col in cursor.fetchall()]
    
    cursor.execute(f"SELECT * FROM [{table_name}] LIMIT 5")
    col_names = [description[0] for description in cursor.description]
    sample_rows = cursor.fetchall()
    
    conn.close()
    
    return {
        "table_name": table_name,
        "row_count": row_count,
        "columns": columns,
        "column_names": col_names,
        "sample_data": [dict(zip(col_names, row)) for row in sample_rows]
    }

def build_context(selected_dataset: str) -> str:
    """Build context string for LLM"""
    dataset_info = get_dataset_summary(selected_dataset)
    
    context = f"""
DATASET INFORMATION:
- Name: {dataset_info['table_name']}
- Total Rows: {dataset_info['row_count']}
- Columns: {', '.join([col['name'] for col in dataset_info['columns']])}

SAMPLE DATA (first 3 rows):
"""
    for i, row in enumerate(dataset_info['sample_data'][:3]):
        context += f"\nRow {i+1}: {json.dumps(row, indent=2)}"
    
    return context

def detect_intent(prompt: str, conversation_history: List[Dict]) -> str:
    """
    Detect user intent:
    - ANALYTICAL_QUESTION: User asking a specific data question (who, what, which, how many, etc.)
    - DASHBOARD_REQUEST: User wants to build a dashboard/visualization
    - GENERAL_CHAT: General conversation
    """
    prompt_lower = prompt.lower()
    
    # Check conversation history first for context
    if conversation_history:
        # Look at last 2 messages
        for msg in conversation_history[-2:]:
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                # If we're already in dashboard conversation, stay in it
                if any(keyword in content for keyword in ["dashboard", "visualization", "insights", "trends", "highlight"]):
                    return "DASHBOARD_REQUEST"
                # If we suggested options, definitely dashboard mode
                if "option" in content or "suggest" in content:
                    return "DASHBOARD_REQUEST"
    
    # Dashboard/visualization keywords - check these first
    dashboard_keywords = ["build", "create", "make", "dashboard", "chart", "visualization", 
                         "graph", "visualize", "design", "looking for", "interested in",
                         "want to see", "i want", "i am looking"]
    
    # Strong indicators of wanting dashboard/analysis (not just a quick question)
    dashboard_phrases = ["i am looking for", "i want to see", "interested in", "focusing on",
                        "particularly interested", "looking to"]
    
    # Check for dashboard intent
    if any(keyword in prompt_lower for keyword in dashboard_keywords):
        return "DASHBOARD_REQUEST"
    
    # Check for phrases that indicate planning/exploration (dashboard mode)
    if any(phrase in prompt_lower for phrase in dashboard_phrases):
        return "DASHBOARD_REQUEST"
    
    # Analytical question keywords - these are QUICK questions
    question_keywords = ["can you show", "show me the", "what is the", "which product has",
                        "how many products", "tell me which", "find the", "list the"]
    
    # Check for direct analytical questions (quick answers)
    if any(keyword in prompt_lower for keyword in question_keywords):
        # Make sure it's a direct question, not a dashboard request
        if not any(keyword in prompt_lower for keyword in dashboard_keywords):
            return "ANALYTICAL_QUESTION"
    
    return "GENERAL_CHAT"

def determine_dashboard_phase(conversation_history: List[Dict], user_prompt: str) -> str:
    """
    For dashboard building mode, determine phase:
    - INTERVIEW: Understanding requirements
    - ITERATION: Suggesting visualization options
    - READY_TO_BUILD: User selected an option
    """
    prompt_lower = user_prompt.lower()
    
    # Check for explicit option selection (READY_TO_BUILD)
    selection_keywords = ["option 1", "option 2", "option 3", "first one", "second one", 
                         "third one", "i'll take", "i want option", "let's go with", 
                         "use that", "sounds good", "perfect"]
    
    if any(keyword in prompt_lower for keyword in selection_keywords):
        if conversation_history:
            for msg in reversed(conversation_history[-4:]):
                if msg.get("role") == "assistant":
                    if "option" in msg.get("content", "").lower():
                        return "READY_TO_BUILD"
    
    # Check if we've already suggested options
    if conversation_history:
        last_assistant_msg = ""
        for msg in reversed(conversation_history):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg.get("content", "").lower()
                break
        
        if "option" in last_assistant_msg and ("chart" in last_assistant_msg or "visualization" in last_assistant_msg):
            return "ITERATION"
    
    # AGGRESSIVE CHECK: User is answering questions with specific details
    # Look for patterns like: "product X, metric Y, audience Z, please suggest"
    answering_with_details_keywords = [
        ("suggest", ["options", "visuali", "chart"]),  # "please suggest options"
        ("walmart", ["retail", "account", "channel"]),   # Mentioned retail
        ("financial analyst", ["audience", "team"]),     # Mentioned audience
        ("sales", ["metric", "dollar", "unit"]),         # Mentioned metrics
        ("2023", ["year", "period", "time"]),            # Mentioned timeframe
        ("pack count", ["product", "pack", "size"]),     # Mentioned filters
    ]
    
    # Count how many of these detail-answer patterns are in the current message
    detail_patterns_found = 0
    for keyword, context_words in answering_with_details_keywords:
        if keyword in prompt_lower:
            # Check if it's in a context that suggests they're answering
            if any(ctx in prompt_lower for ctx in context_words):
                detail_patterns_found += 1
    
    # If they used "suggest" + specific details, they're ready for ITERATION
    if "suggest" in prompt_lower and detail_patterns_found >= 2:
        print(f"DEBUG: User requesting suggestions with {detail_patterns_found} detail patterns - ITERATION")
        return "ITERATION"
    
    # Check for the phrase "please suggest" which is a clear signal
    if any(phrase in prompt_lower for phrase in ["please suggest", "suggest some", "can you suggest"]):
        return "ITERATION"
    
    # If they've provided 3+ specific pieces of information (walmart, sales, audience, etc.)
    specific_answers = [
        "walmart", "albertsons", "kroger", "target",  # Retailers
        "financial", "analyst", "executive", "manager",  # Audiences
        "sales", "units", "dollars", "revenue",  # Metrics
        "2023", "2024", "2022", "q1", "q2", "q3", "q4",  # Time periods
        "pack count", "pack size", "flavor", "brand",  # Product details
        "conventional", "food", "channel",  # Channels
        "cannib", "growth", "trend", "performance"  # Analysis types
    ]
    
    details_found = sum(1 for detail in specific_answers if detail in prompt_lower)
    
    # If 4+ specific details mentioned AND they haven't just asked a basic question, move to ITERATION
    if details_found >= 4 and len(prompt_lower) > 50:  # Longer message = more details provided
        print(f"DEBUG: Found {details_found} specific details in user response - ITERATION")
        return "ITERATION"
    
    return "INTERVIEW"


def answer_analytical_question(prompt: str, selected_dataset: str, conversation_history: List[Dict]) -> Dict[str, Any]:
    """
    Answer direct analytical questions by generating SQL and returning insights
    """
    dataset_context = build_context(selected_dataset)
    
    system_instructions = f"""You are Aevah, a conversational data analyst. The user is asking you an analytical question about their data.

{dataset_context}

YOUR TASK:
1. Generate a SQL query to answer the user's question
2. Return the SQL query in JSON format

CRITICAL SQL RULES:
- ALWAYS wrap column names in square brackets: [Column Name]
- ALWAYS wrap table names in square brackets: [{selected_dataset}]
- Use brackets everywhere: SELECT, WHERE, GROUP BY, ORDER BY, calculations
- For dates, use simple comparison: [Time Period End Date] LIKE '2023%' for year 2023
- For cannibalization: typically [Base Dollars] - [Incr Dollars]
- Use LIMIT to restrict results when appropriate (e.g., "top 5" means LIMIT 5)
- DO NOT use functions like YEAR() - they may not be supported
- Keep queries simple and direct

RESPONSE FORMAT (JSON only):
{{
  "sql_query": "SELECT [Column1], [Column2] FROM [{selected_dataset}] WHERE ... ORDER BY ... LIMIT ...",
  "explanation": "Brief explanation of what this query does"
}}

Examples:
- "Products from 2023 with pack count 12" → SELECT [Product], ([Base Dollars] - [Incr Dollars]) AS Cannibalization FROM [{selected_dataset}] WHERE [Time Period End Date] LIKE '2023%' AND [Pack Count] = 12 ORDER BY Cannibalization DESC
- "How many products with pack count 12?" → SELECT COUNT(*) as count FROM [{selected_dataset}] WHERE [Pack Count] = 12
- "Top 5 products by base dollars" → SELECT [Product], [Base Dollars] FROM [{selected_dataset}] ORDER BY [Base Dollars] DESC LIMIT 5

User Question: {prompt}

Respond with ONLY the JSON, no other text."""

    messages = []
    
    # Add recent conversation for context
    for msg in conversation_history[-4:]:
        if msg.get("role") in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg.get("content", "")})
    
    messages.append({"role": "user", "content": system_instructions})
    
    try:
        payload = {
            "model": "mistralai/mistral-7b-instruct-v0.3",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 500,
            "stream": False
        }
        
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        
        # Parse JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        
        parsed = json.loads(content)
        return parsed
        
    except Exception as e:
        raise Exception(f"Failed to generate SQL: {str(e)}")


def build_dashboard_conversation(prompt: str, selected_dataset: str, conversation_history: List[Dict], phase: str) -> Dict[str, Any]:
    """
    Handle dashboard building conversation (interview → iteration → ready)
    """
    dataset_context = build_context(selected_dataset)
    
    if phase == "INTERVIEW":
        system_instructions = f"""You are Aevah, an AI data analyst helping design a dashboard.

{dataset_context}

The user wants to build a dashboard/visualization. Your role in the INTERVIEW phase:
1. Ask clarifying questions to understand their specific needs
2. Learn about their goals, metrics, and target audience
3. Understand what data they want to visualize

IMPORTANT RULES:
- Ask 1-2 questions at a time, not more
- Listen to their answers and don't repeat previous questions
- If they say "you decide" or similar, STOP ASKING and MOVE TO MAKING RECOMMENDATIONS
- Once you have enough info (goals, metrics, audience, data filters), SUGGEST VISUALIZATION OPTIONS instead of asking more questions

CONVERSATION HISTORY:
{json.dumps(conversation_history[-4:], indent=2)}

Current User Message: {prompt}

If you have enough information about:
- What they want to visualize (e.g., cannibalization rates, product comparison)
- Key filters/data (time period, products, channels, etc.)
- Target audience (who will use this)
- Approximate requirements

Then STOP asking questions and provide 2-3 visualization suggestions.

If the user says "you decide" about ANY detail, that means they TRUST YOUR JUDGMENT - make the decision and move forward with suggestions.

Suggest visualization options ONLY if you have collected enough details. Otherwise ask your question."""

    elif phase == "ITERATION":
        system_instructions = f"""You are Aevah, suggesting visualization options for a dashboard.

{dataset_context}

Based on the user's requirements, suggest 2-3 SPECIFIC visualization options that will help them analyze cannibalization rates, product comparisons, and revenue insights.

Their requirements seem to include:
- Cannibalization rates of products
- Focus on specific pack sizes and flavors
- Comparison within same product universe and retailers (Walmart)
- Target audience: Financial analysts
- Detailed dashboard view
- Metrics: Total sales, revenue, product performance
- Time period: 2023
- Channel: CONVENTIONAL|FOOD

Suggest visualizations that would help them:
1. Compare products and identify cannibalization
2. See trends and patterns
3. Make optimization decisions

Format each option like this:

**Option 1: [Chart Type] - [Title]**
This visualization would display [what it shows] by [dimensions], allowing your financial analysts to [specific business value]. You could easily identify [specific insight], which will help you [decision/action].

**Option 2: [Chart Type] - [Title]**
This visualization would display [what it shows] by [dimensions], allowing your financial analysts to [specific business value]. You could easily identify [specific insight], which will help you [decision/action].

**Option 3: [Chart Type] - [Title]**
This visualization would display [what it shows] by [dimensions], allowing your financial analysts to [specific business value]. You could easily identify [specific insight], which will help you [decision/action].

After suggesting options, ask: "Which of these options would be most useful for your analysis, or would you like me to suggest different approaches?"

DO NOT generate SQL. Just describe the visualization approaches and their benefits."""

    else:  # READY_TO_BUILD
        system_instructions = f"""You are Aevah. The user has selected a visualization option.

{dataset_context}

Your job now is to:
1. Confirm their selection
2. Generate the SQL query to fetch the data
3. Provide the Superset configuration parameters
4. Explain what Step 3 will do

Based on the conversation, the user wants to visualize: cannibalization rates from July 2023, from Walmart, with sales as the metric.

Generate a realistic SQL query and Superset config that would work for their selected visualization.

RESPONSE FORMAT (JSON):
{{
  "confirmation": "Perfect! You've selected [Chart Type]...",
  "sql_query": "SELECT [Column1], [Column2] FROM [{selected_dataset}] WHERE ...",
  "chart_type": "stacked_bar|line|pie|bar|scatter",
  "superset_config": {{
    "title": "Chart Title",
    "x_axis": "Column name for X axis",
    "y_axis": "Column name for Y axis",
    "filters": [
      {{"column": "Column Name", "operator": "==", "value": "Filter Value"}},
      {{"column": "Another Column", "operator": ">=", "value": "Date or Number"}}
    ],
    "group_by": ["Column1", "Column2"],
    "order_by": "Column Name",
    "limit": 100
  }},
  "explanation": "This configuration will create a [Chart Type] showing [what it shows]."
}}

CRITICAL SQL RULES:
- ALWAYS wrap column names in square brackets: [Column Name]
- ALWAYS wrap table names in square brackets: [{selected_dataset}]
- For dates in July 2023, use: [Date Column] LIKE '2023-07%'
- For Walmart filter: WHERE [Retail Account] = 'WALMART' or similar
- Sales metric: likely [Base Dollars] or [Sales] or [Revenue]

User's requirements from conversation:
- Cannibalization rates of products
- July 2023 only
- Walmart only
- Sales as key metric
- Selected: Stacked Bar Chart

Generate only JSON, no other text."""

    messages = []
    for msg in conversation_history[-6:]:
        if msg.get("role") in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg.get("content", "")})
    
    messages.append({"role": "user", "content": f"{system_instructions}\n\nUser: {prompt}"})
    
    try:
        payload = {
            "model": "mistralai/mistral-7b-instruct-v0.3",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800,
            "stream": False
        }
        
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        
        return {
            "response": content,
            "phase": phase
        }
        
    except Exception as e:
        raise Exception(f"Dashboard conversation failed: {str(e)}")


def fix_sql_column_names(query: str, table_name: str) -> str:
    """Add square brackets to unquoted column/table names"""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info([{table_name}])")
        actual_columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        
        # Sort by length (longest first) to avoid partial matches
        actual_columns.sort(key=len, reverse=True)
        
        # First pass: protect already bracketed content
        # Replace [Column] with a placeholder
        protected = []
        def protect_brackets(match):
            protected.append(match.group(1))
            return f"__PROTECTED_{len(protected)-1}__"
        
        query = re.sub(r'\[([^\]]+)\]', protect_brackets, query)
        
        # Second pass: add brackets to unprotected column names
        for col in actual_columns:
            # Match the column name as a whole word, not already protected
            pattern = rf'\b{re.escape(col)}\b(?!__)'
            query = re.sub(pattern, f'[{col}]', query, flags=re.IGNORECASE)
        
        # Third pass: restore protected content
        for i, original in enumerate(protected):
            query = query.replace(f"__PROTECTED_{i}__", f"[{original}]")
        
        # Clean up any double-bracketing
        query = re.sub(r'\[\[([^\]]+)\]\]', r'[\1]', query)
        
        # Ensure table name is bracketed
        query = re.sub(rf'\bFROM\s+{re.escape(table_name)}\b', f'FROM [{table_name}]', query, flags=re.IGNORECASE)
        
        return query
    except Exception as e:
        print(f"Warning: Could not fix SQL query: {e}")
        return query


def execute_sql(query: str, table_name: str = None) -> List[Dict[str, Any]]:
    try:
        if not table_name:
            match = re.search(r'FROM\s+\[?(\w+)\]?', query, re.IGNORECASE)
            if match:
                table_name = match.group(1)
        
        if table_name:
            fixed_query = fix_sql_column_names(query, table_name)
            print(f"Original query: {query}")
            print(f"Fixed query: {fixed_query}")
        else:
            fixed_query = query
        
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(fixed_query)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        results = [dict(zip(columns, r)) for r in rows]
        conn.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL execution error: {str(e)}")


def format_analytical_response(data: List[Dict], sql_query: str, prompt: str, explanation: str) -> str:
    """
    Format the SQL results into a natural language response
    """
    if not data:
        return "I ran the query but didn't find any matching data. Would you like to adjust the criteria?"
    
    # Build a conversational response
    response = f"{explanation}\n\n"
    
    # For single value results
    if len(data) == 1 and len(data[0]) == 1:
        key = list(data[0].keys())[0]
        value = data[0][key]
        response += f"**Answer:** {value}"
    
    # For single row with multiple columns
    elif len(data) == 1:
        response += "**Result:**\n"
        for key, value in data[0].items():
            response += f"- {key}: {value}\n"
    
    # For multiple rows (show as table or list)
    else:
        response += f"**Found {len(data)} results:**\n\n"
        if len(data) <= 10:
            # Show all if 10 or fewer
            for i, row in enumerate(data, 1):
                if len(row) <= 3:
                    # Compact format for few columns
                    row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
                    response += f"{i}. {row_str}\n"
                else:
                    # Detailed format for many columns
                    response += f"\n**#{i}:**\n"
                    for key, value in row.items():
                        response += f"  - {key}: {value}\n"
        else:
            # Show top 5 if more than 10
            response += "(Showing top 5)\n"
            for i, row in enumerate(data[:5], 1):
                row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
                response += f"{i}. {row_str}\n"
            response += f"\n...and {len(data) - 5} more results."
    
    return response


@app.post("/query")
async def process_query(request: QueryRequest):
    """
    Main query endpoint - supports both analytical questions and dashboard building
    """
    try:
        selected = request.selected_dataset
        
        # Auto-select dataset if only one exists
        if not selected:
            table_info = get_available_tables()
            if len(table_info) == 1:
                selected = list(table_info.keys())[0]
            elif len(table_info) == 0:
                return {
                    "type": "chat",
                    "response": "No datasets available. Please upload a CSV file first.",
                    "success": False
                }
        
        conversation_history = request.conversation_history if request.conversation_history else []
        conversation_state = request.conversation_state if request.conversation_state else {}
        
        # Detect intent
        intent = detect_intent(request.prompt, conversation_history)
        
        # Handle based on intent
        if intent == "ANALYTICAL_QUESTION":
            # User is asking a data question - answer it directly
            try:
                sql_result = answer_analytical_question(request.prompt, selected, conversation_history)
                sql_query = sql_result.get("sql_query")
                explanation = sql_result.get("explanation", "Here's what I found:")
                
                # Execute the query
                data = execute_sql(sql_query, selected)
                
                # Format as conversational response
                response_text = format_analytical_response(data, sql_query, request.prompt, explanation)
                
                return {
                    "type": "analytical_answer",
                    "prompt": request.prompt,
                    "response": response_text,
                    "sql_query": sql_query,
                    "data": data,
                    "intent": "ANALYTICAL_QUESTION",
                    "success": True
                }
                
            except Exception as e:
                return {
                    "type": "chat",
                    "response": f"I had trouble answering that question. Could you rephrase it? (Error: {str(e)})",
                    "success": False
                }
        
        elif intent == "DASHBOARD_REQUEST":
            # User wants to build a dashboard - follow interview process
            phase = determine_dashboard_phase(conversation_history, request.prompt)
            conversation_state["phase"] = phase
            
            result = build_dashboard_conversation(request.prompt, selected, conversation_history, phase)
            
            # For READY_TO_BUILD phase, try to parse SQL and config from response
            if phase == "READY_TO_BUILD":
                try:
                    response_content = result["response"]
                    
                    # Try to parse JSON from response
                    if "```json" in response_content:
                        json_str = response_content.split("```json")[1].split("```")[0].strip()
                    elif "{" in response_content:
                        # Find JSON in the response
                        start = response_content.find("{")
                        end = response_content.rfind("}") + 1
                        json_str = response_content[start:end]
                    else:
                        json_str = None
                    
                    if json_str:
                        parsed = json.loads(json_str)
                        
                        return {
                            "type": "ready_to_build",
                            "prompt": request.prompt,
                            "confirmation": parsed.get("confirmation", ""),
                            "sql_query": parsed.get("sql_query", ""),
                            "chart_type": parsed.get("chart_type", "table"),
                            "superset_config": parsed.get("superset_config", {}),
                            "explanation": parsed.get("explanation", ""),
                            "phase": "READY_TO_BUILD",
                            "conversation_state": conversation_state,
                            "intent": "DASHBOARD_REQUEST",
                            "success": True
                        }
                except Exception as e:
                    print(f"Error parsing READY_TO_BUILD response: {e}")
            
            return {
                "type": "dashboard_building",
                "prompt": request.prompt,
                "response": result["response"],
                "phase": phase,
                "conversation_state": conversation_state,
                "intent": "DASHBOARD_REQUEST",
                "success": True
            }
        
        else:
            # General conversation
            return {
                "type": "chat",
                "response": "I'm Aevah, your data analyst assistant. I can help you:\n\n1. Answer questions about your data (e.g., 'Which product has the highest sales?')\n2. Build dashboards and visualizations (e.g., 'Create a chart showing trends')\n\nWhat would you like to explore?",
                "intent": "GENERAL_CHAT",
                "success": True
            }
        
    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), table_name: str = Form(None)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    name = table_name or os.path.splitext(file.filename)[0]
    name = re.sub(r'[^0-9a-zA-Z_]', '_', name)
    conn = get_conn()
    df.to_sql(name, conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    return {"success": True, "table": name, "rows": len(df)}

@app.get("/tables")
async def list_tables():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    rows = cursor.fetchall()
    conn.close()
    return {"tables": [r[0] for r in rows]}

@app.get("/")
async def root():
    return {"message": "Aevah - Conversational Data Analyst + Dashboard Builder API"}

# Saved Charts endpoints
@app.post("/save-chart")
async def save_chart(chart_data: Dict[str, Any]):
    try:
        import uuid
        from datetime import datetime
        
        chart_id = str(uuid.uuid4())
        conn = get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS saved_charts (
                chart_id TEXT PRIMARY KEY,
                title TEXT,
                chart_type TEXT,
                chart_config TEXT,
                data TEXT,
                sql_query TEXT,
                created_at TEXT
            )
        """)
        
        cursor.execute("""
            INSERT INTO saved_charts (chart_id, title, chart_type, chart_config, data, sql_query, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            chart_id,
            chart_data.get('title'),
            chart_data.get('chart_type'),
            json.dumps(chart_data.get('chart_config')),
            json.dumps(chart_data.get('data')),
            chart_data.get('sql_query'),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return {"success": True, "chart_id": chart_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving chart: {str(e)}")

@app.get("/saved-charts")
async def get_saved_charts():
    try:
        conn = get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS saved_charts (
                chart_id TEXT PRIMARY KEY,
                title TEXT,
                chart_type TEXT,
                chart_config TEXT,
                data TEXT,
                sql_query TEXT,
                created_at TEXT
            )
        """)
        
        cursor.execute("SELECT chart_id, title, chart_type, sql_query, created_at FROM saved_charts ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        saved_charts = []
        for row in rows:
            saved_charts.append({
                "chart_id": row[0],
                "title": row[1],
                "chart_type": row[2],
                "sql_query": row[3],
                "created_at": row[4]
            })
        
        return {"saved_charts": saved_charts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading saved charts: {str(e)}")

@app.get("/saved-charts/{chart_id}")
async def get_saved_chart(chart_id: str):
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM saved_charts WHERE chart_id = ?", (chart_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Chart not found")
        
        return {
            "chart_id": row[0],
            "title": row[1],
            "chart_type": row[2],
            "chart_config": json.loads(row[3]),
            "data": json.loads(row[4]),
            "sql_query": row[5],
            "created_at": row[6]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading chart: {str(e)}")

@app.delete("/saved-charts/{chart_id}")
async def delete_saved_chart(chart_id: str):
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM saved_charts WHERE chart_id = ?", (chart_id,))
        conn.commit()
        conn.close()
        
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chart: {str(e)}")