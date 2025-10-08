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
CACHE_TTL = 60 * 60 * 24  # 24 hours

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

    # Template check
    if normalized in TEMPLATES:
        return TEMPLATES[normalized]
    
    # Exact hash check
    h = hash_prompt(prompt)
    if h in CACHE:
        if now - CACHE[h]['time'] < CACHE_TTL:
            return CACHE[h]['response']
        else:
            del CACHE[h]

    # Similarity check with past cached prompts
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

def smart_llm_request(prompt: str, llm_call_func):
    # 1. Try cache/templates
    cached = get_cached_response(prompt)
    if cached:
        return {"type": "chat", "response": cached}

    # 2. Otherwise call LLM
    response = llm_call_func(prompt)

    # 3. Save to cache (only chat responses for now)
    if isinstance(response, dict) and response.get("type") == "chat":
        save_to_cache(prompt, response["response"])
    elif isinstance(response, str):
        save_to_cache(prompt, response)

    return response


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
    conversation_history: Optional[List[Dict]] = []
    selected_dataset: Optional[str] = None

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

def get_table_sample_data(table_name: str, limit: int = 1) -> List[Dict]:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM [{table_name}] LIMIT {limit}")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    results = [dict(zip(columns, r)) for r in rows]
    conn.close()
    return results

def is_data_query(prompt: str) -> bool:
    # Visualization/chart keywords
    viz_keywords = ["show", "display", "chart", "graph", "plot", "visualize", "draw"]
    
    # Summary/insight keywords should NOT trigger SQL mode
    summary_keywords = ["tell me about", "summarize", "summary", "insights", "what can you", "describe", "explain"]
    
    prompt_lower = prompt.lower()
    
    # If asking for summary/insights, don't treat as data query
    if any(keyword in prompt_lower for keyword in summary_keywords):
        return False
    
    # Only treat as data query if asking for visualization
    return any(keyword in prompt_lower for keyword in viz_keywords)

def build_context(selected_dataset: str) -> str:
    table_info = get_available_tables()
    if selected_dataset not in table_info:
        raise HTTPException(status_code=400, detail=f"Dataset '{selected_dataset}' not found. Please select a valid dataset.")

    columns = [col['name'] for col in table_info[selected_dataset]]
    sample_data = get_table_sample_data(selected_dataset, limit=1)
    sample_row = sample_data[0] if sample_data else {}

    sample_str = ", ".join([f"{k}={v}" for k, v in sample_row.items()]) if sample_row else "empty"
    context = f"Dataset={selected_dataset} | Columns={', '.join(columns)} | SampleRow={sample_str}"
    return context


def chat_with_llm(prompt: str, conversation_history: List[Dict] = None, selected_dataset: str = None) -> Dict[str, Any]:
    def llm_call(p: str):
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                if is_data_query(p):
                    # This is a visualization request - generate SQL and chart config
                    if not selected_dataset:
                        raise HTTPException(status_code=400, detail="No dataset selected. Please choose a dataset before asking data-related questions.")
    
                    context = build_context(selected_dataset)
                    system_prompt = f"""You are a data analysis assistant. You MUST respond in valid JSON format only.
    
    Available dataset: {context}
    
    Your response MUST follow this exact structure:
    {{
      "type": "data_query",
      "sql_query": "SELECT [column1], [column2] FROM [{selected_dataset}] WHERE ...",
      "explanation": "Brief explanation of the query",
      "chart_type": "bar|line|pie|scatter|sunburst|table",
      "chart_config": {{
        "title": "Chart Title",
        "x_axis": "column_name_for_x",
        "y_axis": "column_name_for_y"
      }}
    }}
    
    For SUNBURST charts with hierarchical data, use this structure instead:
    {{
      "type": "data_query",
      "sql_query": "SELECT [Company], [Department], COUNT(*) as employee_count FROM [{selected_dataset}] GROUP BY [Company], [Department]",
      "explanation": "Shows employee distribution across companies and departments",
      "chart_type": "sunburst",
      "chart_config": {{
        "title": "Employee Distribution",
        "category_columns": ["Company", "Department"],
        "value_column": "employee_count"
      }}
    }}
    
    CRITICAL SQL RULES - READ CAREFULLY:
    1. ALWAYS wrap column names in square brackets: [Column Name]
    2. ALWAYS wrap table names in square brackets: [{selected_dataset}]
    3. This is REQUIRED for ALL columns, especially those with spaces like [Base Dollars], [Product Universe]
    4. Use brackets in ALL contexts: SELECT, WHERE, GROUP BY, ORDER BY, arithmetic operations
    5. EVERY column reference must have brackets, even in calculations and expressions
    6. Examples of CORRECT syntax:
       - SELECT [Retail Account], [Base Dollars] - [Incr Dollars] AS Cannibalization FROM [{selected_dataset}]
       - SELECT [Product Universe], SUM([Base Dollars]) FROM [{selected_dataset}] GROUP BY [Product Universe]
       - SELECT [Column A], [Column B] * 100 FROM [{selected_dataset}] WHERE [Column C] > 0
    7. Examples of WRONG syntax (DO NOT DO THIS):
       - SELECT Retail Account, Base Dollars FROM Built_Data
       - SELECT Product Universe, Base Dollars - Incr Dollars FROM Built_Data
    
    CHART CONFIG RULES:
    1. ALWAYS include chart_config with proper fields based on chart type
    2. Only use columns that exist in the dataset: {context}
    3. For bar/line/pie/scatter charts: use x_axis and y_axis
    4. For sunburst charts: use category_columns (array) and value_column
    5. CRITICAL: Column names in chart_config must NOT have brackets - use plain names only
       - CORRECT: "x_axis": "Retail Account"
       - WRONG: "x_axis": "[Retail Account]"
    6. If using AS aliases in SQL, use the alias name in chart_config (e.g., if "AS Cannibalization", use "Cannibalization")
    7. sql_query must be valid SQLite syntax using table name: [{selected_dataset}]
    8. NO explanatory text outside the JSON
    
    Example of correct format:
    {{
      "sql_query": "SELECT [Retail Account], [Base Dollars] - [Incr Dollars] AS Cannibalization FROM [{selected_dataset}]",
      "chart_config": {{
        "title": "Cannibalization Analysis",
        "x_axis": "Retail Account",
        "y_axis": "Cannibalization"
      }}
    }}"""
    
                    messages = [
                        {"role": "user", "content": f"{system_prompt}\n\nUser request: {p.strip()}"}
                    ]
                    payload = {
                        "model": "mistralai/mistral-7b-instruct-v0.3",
                        "messages": messages,
                        "temperature": 0.2,
                        "max_tokens": 1000,  # Increased from 500
                        "stream": False
                    }
                else:
                    # Analytical/summary request
                    if selected_dataset:
                        context = build_context(selected_dataset)
                        conn = get_conn()
                        cursor = conn.cursor()
                        
                        cursor.execute(f"SELECT COUNT(*) FROM [{selected_dataset}]")
                        row_count = cursor.fetchone()[0]
                        
                        cursor.execute(f"SELECT * FROM [{selected_dataset}] LIMIT 5")
                        columns = [description[0] for description in cursor.description]
                        sample_rows = cursor.fetchall()
                        conn.close()
                        
                        data_context = f"""Dataset: {selected_dataset}
    Total rows: {row_count}
    Columns: {', '.join(columns)}
    Sample data (first 3 rows):
    """
                        for i, row in enumerate(sample_rows[:3]):
                            data_context += f"\nRow {i+1}: {dict(zip(columns, row))}"
                        
                        enhanced_prompt = f"""You are an AI data analyst. Here's the dataset information:
    
    {data_context}
    
    User question: {p.strip()}
    
    Your job is to:
    1. Provide ACTUAL INSIGHTS and analysis, not just echo back statistics
    2. Look for patterns, trends, and interesting observations in the data
    3. Explain what the data MEANS in a conversational, helpful way
    4. Suggest what analyses or visualizations might be interesting
    5. Be insightful and helpful, not just a number repeater
    
    IMPORTANT: Do NOT just list numbers. Tell the user what's INTERESTING about their data. Make observations, suggest hypotheses, identify patterns.
    
    Respond naturally in plain text - no JSON, just helpful conversation."""
    
                        messages = [
                            {"role": "user", "content": enhanced_prompt}
                        ]
                    else:
                        # General conversation
                        messages = [{"role": "user", "content": p.strip()}]
                    
                    payload = {
                        "model": "mistralai/mistral-7b-instruct-v0.3",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 800  # Increased from 400
                    }
    
                response = requests.post(LM_STUDIO_URL, json=payload, timeout=60)  # Increased from 30 to 60 seconds
                response.raise_for_status()
                data = response.json()
                
                # Check if response was cut off due to max_tokens
                finish_reason = data["choices"][0].get("finish_reason", "")
                content = data["choices"][0]["message"]["content"].strip()
                
                if finish_reason == "length":
                    # Response was truncated - try to handle it gracefully
                    if is_data_query(p):
                        # For data queries, we need complete JSON, so return an error
                        return {"type": "chat", "response": "The query was too complex and the response was cut off. Please try asking for something more specific or simpler."}
                    # For chat, the truncated response might still be useful
                    content += "\n\n[Note: Response was truncated due to length. Please ask for clarification if needed.]"
    
                if is_data_query(p):
                    try:
                        if "```json" in content:
                            content = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content:
                            content = content.split("```")[1].strip()
                        parsed = json.loads(content)
                        
                        if "sql_query" not in parsed:
                            raise ValueError("Missing sql_query in LLM response")
                        if "chart_config" not in parsed or not isinstance(parsed["chart_config"], dict):
                            raise ValueError("Missing or invalid chart_config in LLM response")
                        if "title" not in parsed["chart_config"]:
                            raise ValueError("Missing title in chart_config")
                        
                        return parsed
                    except Exception as e:
                        return {"type": "chat", "response": f"I couldn't generate a proper data query. Error: {str(e)}. Please try rephrasing your question."}
                else:
                    return {"type": "chat", "response": content}
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"Timeout on attempt {attempt + 1}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    return {"type": "chat", "response": f"The AI is taking too long to respond. This might be because:\n1. The model is processing a complex query\n2. LM Studio server is overloaded\n3. Your computer needs more resources\n\nPlease try:\n- Asking a simpler question\n- Restarting LM Studio\n- Waiting a moment and trying again"}
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Request error on attempt {attempt + 1}: {e}, retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"LM Studio connection error after {max_retries} attempts: {str(e)}")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Optimization wrapper: cache, templates, similarity check
    return smart_llm_request(prompt, llm_call)


def fix_sql_column_names(query: str, table_name: str) -> str:
    """Add square brackets to unquoted column/table names and aliases"""
    try:
        # Get actual column names from the table
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info([{table_name}])")
        actual_columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        
        # Sort columns by length (longest first) to avoid partial matches
        actual_columns.sort(key=len, reverse=True)
        
        # Quote each column name that appears in the query
        for col in actual_columns:
            # Match the column name not already in brackets
            # Look for word boundaries
            pattern = rf'\b{re.escape(col)}\b(?!\])'
            
            # Replace with bracketed version
            query = re.sub(pattern, f'[{col}]', query, flags=re.IGNORECASE)
        
        # Also ensure table name is bracketed
        query = re.sub(rf'\b{re.escape(table_name)}\b(?!\])', f'[{table_name}]', query, flags=re.IGNORECASE)
        
        # Fix aliases with spaces: "as Total Sales" -> "as [Total Sales]"
        # Match "as <word> <word>..." patterns that aren't already bracketed
        # Use a more greedy pattern that stops at comma, FROM, WHERE, etc.
        alias_pattern = r'\bas\s+(?!\[)([A-Za-z][A-Za-z0-9\s]+?)(?=\s*(?:,|FROM|WHERE|GROUP|ORDER|HAVING|LIMIT|$))'
        
        def bracket_alias(match):
            alias = match.group(1).strip()
            # Only bracket if it contains spaces and isn't already bracketed
            if ' ' in alias and not alias.startswith('['):
                return f'as [{alias}]'
            return match.group(0)
        
        query = re.sub(alias_pattern, bracket_alias, query, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up any double-bracketing [[column]] -> [column]
        query = re.sub(r'\[\[([^\]]+)\]\]', r'[\1]', query)
        
        return query
    except Exception as e:
        # If fixing fails, return original query
        print(f"Warning: Could not fix SQL query: {e}")
        return query

def execute_sql(query: str, table_name: str = None) -> List[Dict[str, Any]]:
    try:
        # Try to extract table name from query if not provided
        if not table_name:
            match = re.search(r'FROM\s+\[?(\w+)\]?', query, re.IGNORECASE)
            if match:
                table_name = match.group(1)
        
        # Fix column names
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
    
def clean_chart_config(chart_config: dict) -> dict:
    """Remove brackets from column names in chart_config"""
    cleaned = {}
    for key, value in chart_config.items():
        if isinstance(value, str):
            # Remove square brackets from column names
            cleaned[key] = value.strip('[]')
        elif isinstance(value, list):
            # Handle arrays (like category_columns for sunburst)
            cleaned[key] = [v.strip('[]') if isinstance(v, str) else v for v in value]
        else:
            cleaned[key] = value
    return cleaned


def normalize_chart_type(chart_type: str) -> str:
    if not chart_type:
        return "table"
    ct = chart_type.strip().lower()
    mapping = {
        "bar chart": "bar",
        "bar": "bar",
        "line chart": "line",
        "line": "line",
        "pie chart": "pie",
        "pie": "pie",
        "scatter": "scatter",
        "scatter plot": "scatter",
        "radar": "radar",
        "sunburst": "sunburst",
        "treemap": "treemap",
        "world_map": "world_map",
        "table": "table",
        "heatmap": "heatmap"
    }
    return mapping.get(ct, "table")


@app.post("/query")
async def process_query(request: QueryRequest):
    # Auto-select dataset if not provided and only one exists
    selected = request.selected_dataset
    auto_selected = False
    
    # Check if this is a data-related question (visualization or analysis)
    if not selected:
        table_info = get_available_tables()
        if len(table_info) == 1:
            selected = list(table_info.keys())[0]
            auto_selected = True
        elif len(table_info) == 0 and (is_data_query(request.prompt) or any(kw in request.prompt.lower() for kw in ["dataset", "data", "analyze", "tell me about"])):
            return {
                "type": "chat",
                "prompt": request.prompt,
                "response": "No datasets available. Please upload a CSV file first before asking data-related questions.",
                "success": False
            }
    
    llm_response = chat_with_llm(request.prompt, request.conversation_history, selected)

    sql_query = llm_response.get("sql_query") if isinstance(llm_response, dict) else None

    if sql_query:
        try:
            data = execute_sql(sql_query, selected)  # Pass table name for fixing
            
            if not data:
                return {
                    "type": "chat",
                    "prompt": request.prompt,
                    "response": "The query executed successfully but returned no data. The dataset might be empty or your filters were too restrictive.",
                    "success": True
                }
            
            chart_type = normalize_chart_type(llm_response.get("chart_type"))
            chart_config = llm_response.get("chart_config", {})
            
            # Clean brackets from chart_config column names
            chart_config = clean_chart_config(chart_config)
            
            # Ensure chart_config has minimum required fields
            if not chart_config.get("title"):
                chart_config["title"] = "Data Analysis"
            
            # For sunburst charts, validate category_columns and value_column
            if chart_type == "sunburst":
                if not chart_config.get("category_columns") or not chart_config.get("value_column"):
                    # Try to infer from data if possible
                    if data and len(data) > 0:
                        columns = list(data[0].keys())
                        # Look for hierarchical columns (non-numeric)
                        text_cols = [col for col in columns if isinstance(data[0][col], str)]
                        numeric_cols = [col for col in columns if isinstance(data[0][col], (int, float))]
                        
                        if len(text_cols) >= 2 and len(numeric_cols) >= 1:
                            chart_config["category_columns"] = text_cols[:2]
                            chart_config["value_column"] = numeric_cols[0]
                        else:
                            chart_type = "table"
            
            # For non-table/sunburst charts, validate required axis fields
            elif chart_type not in ["table", "treemap", "organization"]:
                if not chart_config.get("x_axis") or not chart_config.get("y_axis"):
                    # Try to infer from data
                    if data and len(data) > 0:
                        columns = list(data[0].keys())
                        if len(columns) >= 2:
                            chart_config["x_axis"] = columns[0]
                            chart_config["y_axis"] = columns[1]
                        else:
                            # Fall back to table if we can't infer axes
                            chart_type = "table"
            
            # Add note about auto-selection
            interpretation = llm_response.get("explanation", "Query executed successfully")
            if auto_selected:
                interpretation = f"[Using dataset: {selected}] {interpretation}"
            
            return {
                "type": "data_query",
                "prompt": request.prompt,
                "sql_query": sql_query,
                "interpretation": interpretation,
                "chart_type": chart_type,
                "chart_config": chart_config,
                "data": data,
                "success": True
            }
        except Exception as e:
            return {
                "type": "chat",
                "prompt": request.prompt,
                "response": f"The AI generated an invalid SQL query. Error: {str(e)}. Please try rephrasing your question.",
                "success": False
            }
    else:
        return {
            "type": "chat",
            "prompt": request.prompt,
            "response": llm_response.get("response", str(llm_response)),
            "success": True
        }

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
    return {"message": "Data Analyst Chat API (runs AI SQL and returns data)."}

# Saved Charts endpoints
@app.post("/save-chart")
async def save_chart(chart_data: Dict[str, Any]):
    """Save a chart configuration to the database"""
    try:
        import uuid
        from datetime import datetime
        
        chart_id = str(uuid.uuid4())
        conn = get_conn()
        cursor = conn.cursor()
        
        # Create saved_charts table if it doesn't exist
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
    """Get all saved charts"""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
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
    """Get a specific saved chart"""
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
    """Delete a saved chart"""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM saved_charts WHERE chart_id = ?", (chart_id,))
        conn.commit()
        conn.close()
        
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chart: {str(e)}")