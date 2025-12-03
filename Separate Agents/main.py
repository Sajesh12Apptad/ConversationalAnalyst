"""
Main Orchestrator - Routes requests to appropriate agents
This is the entry point that coordinates all agents
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import os
import re
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import json
import time

# Import agents
from sql_agent import get_sql_agent
from conversation_agent import get_conversation_agent
from visualization_agent import get_visualization_agent
from analysis_agent import get_analysis_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "app_data.db"
CHATS_FILE = "saved_chats.json"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

class QueryRequest(BaseModel):
    prompt: str
    conversation_history: Optional[List[Dict]] = None
    selected_dataset: Optional[str] = None
    conversation_state: Optional[Dict] = None
    chat_id: Optional[str] = None

# ==================== CHAT MANAGEMENT ====================

def load_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_chats(chats):
    with open(CHATS_FILE, 'w') as f:
        json.dump(chats, f, indent=2)

# ==================== MAIN ORCHESTRATOR ====================

class Orchestrator:
    """
    Main orchestrator that routes requests to appropriate agents
    """
    
    def __init__(self):
        self.sql_agent = get_sql_agent()
        self.conversation_agent = get_conversation_agent()
        self.visualization_agent = get_visualization_agent()
        self.analysis_agent = get_analysis_agent()
    
    def process_query(self, request: QueryRequest) -> Dict[str, Any]:
        """Main entry point - routes to appropriate agent based on intent"""
        print(f"\n{'='*60}")
        print(f"[ORCHESTRATOR] Processing query: {request.prompt}")
        print(f"{'='*60}\n")
        
        selected = request.selected_dataset
        if not selected:
            table_info = self._get_available_tables()
            if len(table_info) == 1:
                selected = list(table_info.keys())[0]
            else:
                return {
                    "type": "chat",
                    "response": "Please upload a dataset first, or select one from the dropdown.",
                    "success": True
                }
        
        conversation_history = request.conversation_history or []
        conversation_state = request.conversation_state or {}
        
        print("[ORCHESTRATOR] Step 1: Detecting intent...")
        intent = self.conversation_agent.detect_intent(
            request.prompt,
            conversation_history,
            conversation_state
        )
        print(f"[ORCHESTRATOR] Intent detected: {intent}\n")
        
        try:
            if intent == "ANALYTICAL_QUESTION":
                return self._handle_analytical_question(request.prompt, selected, conversation_state)
            elif intent == "DASHBOARD_REQUEST":
                return self._handle_dashboard_request(request.prompt, selected, conversation_history, conversation_state)
            elif intent == "VISUALIZATION_ANALYSIS":
                return self._handle_visualization_analysis(request.prompt, conversation_state)
            elif intent == "CHART_REFINEMENT":
                return self._handle_chart_refinement(request.prompt, selected, conversation_state)
            elif intent == "DATASET_INFO":
                return self._handle_dataset_info(request.prompt, selected)
            elif intent == "EXECUTIVE_SUMMARY":
                return self._handle_executive_summary(selected)
            elif intent == "FORECASTING":
                return self._handle_forecasting(request.prompt, selected)
            else:
                return self._handle_general_chat(request.prompt, selected, conversation_state)
        except Exception as e:
            print(f"[ORCHESTRATOR] Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "type": "error",
                "response": f"I encountered an error: {str(e)}",
                "success": False
            }
    
    def _handle_analytical_question(self, prompt: str, table_name: str, conversation_state: Dict) -> Dict[str, Any]:
        """Handle analytical data questions - LLM-powered SQL generation"""
        print("[ORCHESTRATOR] Routing to SQL Agent...")
        
        dataset_info = self.sql_agent.get_dataset_info(table_name)
        
        if conversation_state.get("awaiting_clarification"):
            print("[ORCHESTRATOR] User provided clarification")
            original_question = conversation_state.get("original_question", "")
            combined_prompt = f"{original_question}. Additional context: {prompt}"
            conversation_state["awaiting_clarification"] = False
            conversation_state.pop("original_question", None)
            prompt = combined_prompt
        
        # Execute query through SQL agent
        result = self.sql_agent.handle_analytical_query(prompt, table_name, conversation_state)
        
        # Handle error responses
        if result.get("type") == "error" or not result.get("success", True):
            return {
                "type": "analytical_answer",
                "response": result.get("response", "I encountered an error processing your query."),
                "sql_query": result.get("sql_query", ""),
                "data": [],
                "conversation_state": conversation_state,
                "success": False
            }
        
        sql_query = result.get("sql_query") or result.get("sql", "")
        data = result.get("data", [])
        
        print("[ORCHESTRATOR] Routing to Conversation Agent for explanation...")
        explanation = self.conversation_agent.explain_data_results(data, result, prompt)
        
        if result.get("explanation"):
            explanation = f"{result['explanation']}\n\n{explanation}"
        
        conversation_state["last_analytical_query"] = {
            "sql": sql_query,
            "data": data,
            "prompt": prompt
        }
        
        return {
            "type": "analytical_answer",
            "response": explanation,
            "sql_query": sql_query,
            "data": data,
            "x_axis": result.get("x_axis"),
            "y_axis": result.get("y_axis"),
            "chart_type": result.get("chart_type", "bar"),
            "conversation_state": conversation_state,
            "success": True
        }
    
    def _handle_dashboard_request(self, prompt: str, table_name: str, conversation_history: List[Dict], conversation_state: Dict) -> Dict[str, Any]:
        """Handle dashboard/visualization building requests"""
        print("[ORCHESTRATOR] Routing to Visualization Agent...")
        
        phase = self._determine_dashboard_phase(prompt, conversation_state, conversation_history)
        print(f"[ORCHESTRATOR] Dashboard phase: {phase}")
        
        conversation_state["dashboard_phase"] = phase
        
        if phase == "INTERVIEW":
            dataset_info = self.sql_agent.get_dataset_info(table_name)
            response = self.visualization_agent.ask_dashboard_questions(prompt, dataset_info)
            conversation_state["dashboard_phase"] = "GATHERING"
            conversation_state["questions_asked"] = 1
            conversation_state["user_answers"] = []
            
            return {
                "type": "dashboard_building",
                "response": response,
                "phase": "INTERVIEW",
                "conversation_state": conversation_state,
                "success": True
            }
        
        elif phase == "GATHERING":
            conversation_state["user_answers"].append(prompt)
            questions_asked = conversation_state.get("questions_asked", 1)
            
            has_metric = self._check_has_metric(conversation_state["user_answers"])
            has_timeframe = self._check_has_timeframe(conversation_state["user_answers"])
            has_purpose = self._check_has_purpose(conversation_state["user_answers"])
            
            user_was_vague = all(len(answer.split()) < 4 for answer in conversation_state["user_answers"])
            
            if questions_asked >= 2 or not user_was_vague or (has_metric or has_timeframe):
                conversation_state["dashboard_phase"] = "ITERATION"
                phase = "ITERATION"
            else:
                dataset_info = self.sql_agent.get_dataset_info(table_name)
                follow_up = self.visualization_agent.ask_simple_followup(
                    conversation_state["user_answers"],
                    dataset_info
                )
                
                if follow_up:
                    conversation_state["questions_asked"] = questions_asked + 1
                    return {
                        "type": "dashboard_building",
                        "response": follow_up,
                        "phase": "GATHERING",
                        "conversation_state": conversation_state,
                        "success": True
                    }
                else:
                    conversation_state["dashboard_phase"] = "ITERATION"
                    phase = "ITERATION"
        
        if phase == "ITERATION":
            dataset_info = self.sql_agent.get_dataset_info(table_name)
            user_requirements = " ".join(conversation_state.get("user_answers", []))
            
            if not user_requirements:
                user_requirements = " ".join([
                    msg.get("content", "")
                    for msg in conversation_history[-5:]
                    if msg.get("role") == "user"
                ])
            
            recommendations = self.visualization_agent.recommend_visualizations(
                user_requirements,
                dataset_info,
                conversation_history
            )
            
            conversation_state["recommended_visualizations"] = recommendations
            response = self.visualization_agent.format_recommendations_for_user(recommendations)
            
            return {
                "type": "dashboard_building",
                "response": response,
                "phase": "ITERATION",
                "conversation_state": conversation_state,
                "success": True
            }
        
        elif phase == "READY_TO_BUILD":
            return self._build_selected_visualization(prompt, table_name, conversation_state)
    
    def _build_selected_visualization(self, prompt: str, table_name: str, conversation_state: Dict) -> Dict[str, Any]:
        """Build the selected visualization"""
        print("[ORCHESTRATOR] Building selected visualization...")
        
        prompt_lower = prompt.lower()
        option_num = 0
        if "option 1" in prompt_lower or "first" in prompt_lower:
            option_num = 1
        elif "option 2" in prompt_lower or "second" in prompt_lower:
            option_num = 2
        elif "option 3" in prompt_lower or "third" in prompt_lower:
            option_num = 3
        
        recommendations = conversation_state.get("recommended_visualizations", [])
        selected_viz = next((r for r in recommendations if r["option_number"] == option_num), None)
        
        if not selected_viz:
            return {
                "type": "error",
                "response": "I couldn't find that option. Please select Option 1, 2, or 3.",
                "success": False
            }
        
        print("[ORCHESTRATOR] Routing to SQL Agent for data...")
        sql_result = self.sql_agent.handle_analytical_query(
            f"Get data for {selected_viz['title']}",
            table_name,
            {"visualization": selected_viz}
        )
        
        print("[ORCHESTRATOR] Generating Superset configuration...")
        config = self.visualization_agent.generate_superset_config(
            selected_viz,
            sql_result.get('sql_query', ''),
            sql_result.get('data', [])
        )
        
        conversation_state["last_chart_generated"] = {
            "sql_query": sql_result.get('sql_query', ''),
            "chart_type": selected_viz['chart_type'],
            "preview_data": sql_result.get('data', []),
            "config": config
        }
        conversation_state["dashboard_phase"] = None
        
        data = sql_result.get('data', [])
        x_axis = list(data[0].keys())[0] if data else "x"
        y_axis = list(data[0].keys())[1] if data and len(data[0]) > 1 else "y"
        
        return {
            "type": "multi_response",
            "responses": [
                {
                    "type": "chart_preview",
                    "data": data,
                    "chart_type": selected_viz['chart_type'],
                    "chart_config": {
                        "title": selected_viz['title'],
                        "x_axis": x_axis,
                        "y_axis": y_axis
                    }
                },
                {
                    "type": "ready_to_build",
                    "confirmation": f"Perfect! Here's your {selected_viz['chart_type']} visualization.",
                    "config": config,
                    "sql_query": sql_result.get('sql_query', ''),
                    "phase": "READY_TO_BUILD"
                }
            ],
            "conversation_state": conversation_state,
            "success": True
        }
    
    def _handle_visualization_analysis(self, prompt: str, conversation_state: Dict) -> Dict[str, Any]:
        """Handle questions about visualizations"""
        print("[ORCHESTRATOR] Routing to Analysis Agent...")
        
        last_chart = conversation_state.get("last_chart_generated")
        if not last_chart:
            return {
                "type": "error",
                "response": "I don't have a visualization to discuss. Please create a chart first!",
                "success": False
            }
        
        analysis = self.analysis_agent.analyze_visualization(prompt, last_chart, {})
        
        return {
            "type": "visualization_analysis",
            "response": analysis,
            "conversation_state": conversation_state,
            "success": True
        }
    
    def _handle_chart_refinement(self, prompt: str, table_name: str, conversation_state: Dict) -> Dict[str, Any]:
        """Handle chart refinement requests"""
        print("[ORCHESTRATOR] Handling chart refinement...")
        
        last_chart = conversation_state.get("last_chart_generated")
        if not last_chart:
            return {
                "type": "error",
                "response": "I don't have a chart to refine. Please create a visualization first!",
                "success": False
            }
        
        return self._handle_analytical_question(prompt, table_name, conversation_state)
    
    def _handle_dataset_info(self, prompt: str, table_name: str) -> Dict[str, Any]:
        """Handle dataset information requests"""
        print("[ORCHESTRATOR] Providing dataset information...")
        
        dataset_info = self.sql_agent.get_dataset_info(table_name)
        
        response = f"""📊 **About Your Dataset: {dataset_info['table_name']}**

This dataset contains **{dataset_info['row_count']:,} records** with **{len(dataset_info['columns'])} columns**.

**Available Columns:**
{', '.join([col['name'] for col in dataset_info['columns'][:10]])}{"..." if len(dataset_info['columns']) > 10 else ""}

{f"**Date Column:** {dataset_info['date_column']}" if dataset_info.get('date_column') else ""}

What would you like to explore?"""
        
        return {
            "type": "dataset_info",
            "response": response,
            "success": True
        }
    
    def _handle_executive_summary(self, table_name: str) -> Dict[str, Any]:
        """Handle executive summary requests"""
        print("[ORCHESTRATOR] Routing to Analysis Agent for executive summary...")
        
        dataset_info = self.sql_agent.get_dataset_info(table_name)
        summary = self.analysis_agent.generate_executive_summary(dataset_info)
        
        return {
            "type": "executive_summary",
            "response": summary,
            "success": True
        }
    
    def _handle_forecasting(self, prompt: str, table_name: str) -> Dict[str, Any]:
        """Handle forecasting requests"""
        print("[ORCHESTRATOR] Handling forecast request...")
        
        sql_result = self.sql_agent.handle_analytical_query(prompt, table_name, {})
        
        data = sql_result.get('data', [])
        if not data or len(data) < 2:
            return {
                "type": "error",
                "response": "I don't have enough historical data to generate a reliable forecast.",
                "success": False
            }
        
        metric_name = list(data[0].keys())[-1]
        forecast_result = self.analysis_agent.generate_forecast(data, metric_name, periods=3, forecast_unit="month")
        response = self.analysis_agent.format_forecast_response(forecast_result, metric_name, 3, "month")
        
        return {
            "type": "forecast",
            "response": response,
            "success": True
        }
    
    def _handle_general_chat(self, prompt: str, table_name: str, conversation_state: Dict) -> Dict[str, Any]:
        """Handle general conversation"""
        print("[ORCHESTRATOR] Routing to Conversation Agent for general chat...")
        
        context = {"dataset": table_name, "has_data": bool(table_name)}
        response = self.conversation_agent.generate_response(prompt, context)
        
        return {
            "type": "chat",
            "response": response,
            "conversation_state": conversation_state,
            "success": True
        }
    
    def _determine_dashboard_phase(self, prompt: str, conversation_state: Dict, conversation_history: List[Dict]) -> str:
        """Determine which phase of dashboard building we're in"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["option 1", "option 2", "option 3", "first", "second", "third", "i choose", "i select"]):
            return "READY_TO_BUILD"
        
        current_phase = conversation_state.get("dashboard_phase")
        
        if not current_phase:
            return "INTERVIEW"
        
        if current_phase == "GATHERING":
            return "GATHERING"
        
        if current_phase == "ITERATION":
            return "ITERATION"
        
        return current_phase
    
    def _check_has_metric(self, user_answers: List[str]) -> bool:
        combined = " ".join(user_answers).lower()
        metric_keywords = ["sales", "revenue", "dollars", "units", "quantity", "profit", "margin", "performance", "growth", "trend", "volume", "count", "average", "total"]
        return any(keyword in combined for keyword in metric_keywords)
    
    def _check_has_timeframe(self, user_answers: List[str]) -> bool:
        combined = " ".join(user_answers).lower()
        time_keywords = ["2023", "2024", "2022", "year", "month", "quarter", "week", "day", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "q1", "q2", "q3", "q4", "last", "past", "recent", "current", "this"]
        return any(keyword in combined for keyword in time_keywords)
    
    def _check_has_purpose(self, user_answers: List[str]) -> bool:
        combined = " ".join(user_answers).lower()
        purpose_keywords = ["trend", "compare", "comparison", "distribution", "breakdown", "top", "bottom", "highest", "lowest", "performance", "analysis", "insight", "executive", "team", "management"]
        return any(keyword in combined for keyword in purpose_keywords)
    
    def _get_available_tables(self):
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


# Create orchestrator instance
orchestrator = Orchestrator()

# ==================== API ENDPOINTS ====================

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        result = orchestrator.process_query(request)
        return result
    except Exception as e:
        print(f"[API] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
    return {"message": "Aevah - Multi-Agent Conversational Data Analyst API"}

# ==================== CHAT MANAGEMENT ENDPOINTS ====================

@app.post("/chats/create")
async def create_chat():
    try:
        chat_id = str(uuid.uuid4())
        chats = load_chats()
        chats[chat_id] = {
            "id": chat_id,
            "title": "New Chat",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "conversation": [],
            "conversation_state": {},
            "selected_dataset": None
        }
        save_chats(chats)
        return {"success": True, "chat_id": chat_id, "chat": chats[chat_id]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats")
async def list_chats():
    try:
        chats = load_chats()
        chat_list = list(chats.values())
        chat_list.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return {"success": True, "chats": chat_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    try:
        chats = load_chats()
        if chat_id not in chats:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"success": True, "chat": chats[chat_id]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chats/{chat_id}/save")
async def save_chat(chat_id: str, chat_data: Dict[str, Any]):
    try:
        chats = load_chats()
        if chat_id not in chats:
            chats[chat_id] = {
                "id": chat_id,
                "title": chat_data.get("title", "New Chat"),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "conversation": [],
                "conversation_state": {},
                "selected_dataset": None
            }
        chats[chat_id].update({
            "title": chat_data.get("title", chats[chat_id].get("title", "New Chat")),
            "conversation": chat_data.get("conversation", chats[chat_id].get("conversation", [])),
            "conversation_state": chat_data.get("conversation_state", chats[chat_id].get("conversation_state", {})),
            "selected_dataset": chat_data.get("selected_dataset", chats[chat_id].get("selected_dataset")),
            "updated_at": datetime.now().isoformat()
        })
        save_chats(chats)
        return {"success": True, "chat": chats[chat_id]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    try:
        chats = load_chats()
        if chat_id in chats:
            del chats[chat_id]
            save_chats(chats)
        return {"success": True, "message": "Chat deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))