"""
Conversation Agent - Handles natural language interactions and explanations
Focuses on understanding user intent and providing helpful responses
"""

import requests
import json
from typing import List, Dict, Any

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

class ConversationAgent:
    """
    Handles conversational aspects:
    - Intent detection
    - Natural language responses
    - Clarifying questions
    - Explanations
    """
    
    def detect_intent(self, user_prompt: str, conversation_history: List[Dict], conversation_state: Dict) -> str:
        """
        Detect user intent to route to appropriate agent
        This is lightweight - just classification, not deep reasoning
        """
        prompt_lower = user_prompt.lower()
        
        # HIGHEST PRIORITY: Check if we're awaiting clarification
        if conversation_state.get("awaiting_clarification"):
            # User is responding to our clarifying questions
            print(f"DEBUG [Conversation]: User responding to clarification")
            return "ANALYTICAL_QUESTION"
        
        # Check for visualization analysis (user asking about a chart they see)
        if conversation_state.get("last_chart_generated"):
            viz_patterns = [
                "what can you tell me", "what does this show", "what do you see",
                "explain this", "tell me about", "what's happening", "what happened",
                "from the visualization", "from this chart", "interesting", "insights"
            ]
            if any(pattern in prompt_lower for pattern in viz_patterns):
                return "VISUALIZATION_ANALYSIS"
        
        # Check for chart refinement
        if conversation_state.get("last_chart_generated"):
            refinement_patterns = [
                "change the x", "change the y", "use different", "instead of",
                "make it a pie", "make it a bar", "make it a line", "switch to"
            ]
            if any(pattern in prompt_lower for pattern in refinement_patterns):
                return "CHART_REFINEMENT"
        
        # Check for dashboard/visualization building
        dashboard_keywords = [
            "build visualizations", "create visualizations", "build dashboards",
            "create dashboards", "build a dashboard", "i want to visualize",
            "show me a chart", "create a chart", "build a chart", "build a visual",
            "create a visual", "create visual", "build visual", "make a chart",
            "make a visualization", "i want a chart", "i want a visualization",
            "visualize", "visualization", "visual", "chart", "graph"
        ]
        # More flexible matching - check if any dashboard keyword is in the prompt
        if any(keyword in prompt_lower for keyword in dashboard_keywords):
            # Make sure it's not an analytical question with "show me"
            analytical_indicators = ["which", "what is the", "how many", "show me the data", "tell me"]
            if not any(indicator in prompt_lower for indicator in analytical_indicators):
                return "DASHBOARD_REQUEST"
        
        # Check if in active dashboard flow
        if conversation_state.get("dashboard_phase"):
            # Check for option selection
            if any(word in prompt_lower for word in ["option 1", "option 2", "option 3", "first", "second", "third"]):
                return "DASHBOARD_REQUEST"
            # Stay in dashboard flow unless explicit analytical question
            explicit_analytical = ["which product", "what product", "how many", "show me the data"]
            if not any(keyword in prompt_lower for keyword in explicit_analytical):
                return "DASHBOARD_REQUEST"
        
        # Check for forecasting
        forecast_keywords = [
            "forecast", "predict", "prediction", "future", "next month", "next year",
            "projection", "what will happen", "trend"
        ]
        if any(keyword in prompt_lower for keyword in forecast_keywords):
            return "FORECASTING"
        
        # Check for analytical questions (data queries)
        analytical_patterns = [
            "which product", "what product", "show me", "tell me",
            "how many", "how much", "highest", "lowest", "top", "bottom",
            "most", "least", "average", "total", "down in sales", "up in sales",
            "compare", "difference between"
        ]
        is_question = any(q in prompt_lower for q in ["which", "what", "how", "who", "where"])
        has_analytical = any(pattern in prompt_lower for pattern in analytical_patterns)
        
        if is_question and has_analytical:
            return "ANALYTICAL_QUESTION"
        
        # Check for dataset info requests
        if any(keyword in prompt_lower for keyword in ["tell me about the dataset", "describe the dataset", "about the data"]):
            return "DATASET_INFO"
        
        # Check for executive summary
        if any(keyword in prompt_lower for keyword in ["executive summary", "generate report", "comprehensive report"]):
            return "EXECUTIVE_SUMMARY"
        
        # Default to general chat
        return "GENERAL_CHAT"
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Generate natural language response for general conversation
        """
        instruction = f"""You are Aevah, a friendly conversational data analyst.

CONTEXT:
{json.dumps(context, indent=2)}

USER: {prompt}

Respond naturally and helpfully. Be concise but informative."""

        try:
            payload = {
                "model": "mistralai/mistral-7b-instruct-v0.3",
                "messages": [{"role": "user", "content": instruction}],
                "temperature": 0.7,
                "max_tokens": 300,
                "stream": False
            }
            
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            return f"I encountered an error: {str(e)}"
    
    def explain_data_results(self, data: List[Dict], query_context: Dict, user_prompt: str) -> str:
        """
        Generate natural language explanation of data results
        """
        if not data:
            return "I ran the query but didn't find any matching data."
        
        # Build concise explanation
        intent = query_context.get("intent", "")
        comparison = query_context.get("comparison_context", "")
        row_count = len(data)
        
        explanation = f"Based on your question about {intent.lower() if intent else 'the data'}"
        
        if comparison:
            explanation += f" ({comparison})"
        
        explanation += f", here's what I found:\n\n"
        
        # Format results
        if row_count == 1 and len(data[0]) == 1:
            # Single value result
            key = list(data[0].keys())[0]
            value = data[0][key]
            explanation += f"**{key}:** {value}"
        
        elif row_count == 1:
            # Single row, multiple columns
            explanation += "**Result:**\n"
            for key, value in data[0].items():
                explanation += f"- {key}: {value}\n"
        
        elif row_count <= 10:
            # Show all results
            explanation += f"**Found {row_count} results:**\n\n"
            for i, row in enumerate(data, 1):
                row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
                explanation += f"{i}. {row_str}\n"
        
        else:
            # Show top 5
            explanation += f"**Found {row_count} results (showing top 5):**\n\n"
            for i, row in enumerate(data[:5], 1):
                row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
                explanation += f"{i}. {row_str}\n"
            explanation += f"\n...and {row_count - 5} more results."
        
        return explanation
    
    def ask_clarifying_questions(self, user_prompt: str, dataset_info: Dict) -> str:
        """
        Ask clarifying questions when user request is ambiguous
        Especially important for comparisons and "down/up" questions
        """
        prompt_lower = user_prompt.lower()
        
        # Check if asking about trends without time context
        if any(word in prompt_lower for word in ["down", "up", "declining", "increasing", "trend"]):
            if not any(word in prompt_lower for word in ["year", "month", "quarter", "week", "2023", "2024"]):
                return """I'd be happy to help you analyze that! To give you accurate insights, I need a bit more context:

1. **Time Period**: Are you comparing to:
   - Last year (year-over-year)?
   - Last quarter (quarter-over-quarter)?
   - Last month (month-over-month)?
   - A specific time period?

2. **Metric**: Which metric should I analyze?
   - Sales revenue?
   - Units sold?
   - Both?

Let me know and I'll pull the data for you!"""
        
        return None
    
    def format_for_display(self, text: str) -> str:
        """
        Format text for better display in UI
        """
        # Add proper line breaks and formatting
        formatted = text.replace('\n\n', '<br><br>')
        formatted = formatted.replace('\n', '<br>')
        
        # Bold markdown
        formatted = formatted.replace('**', '<strong>').replace('**', '</strong>')
        
        return formatted


# Singleton instance
_conversation_agent = None

def get_conversation_agent() -> ConversationAgent:
    """Get singleton conversation agent instance"""
    global _conversation_agent
    if _conversation_agent is None:
        _conversation_agent = ConversationAgent()
    return _conversation_agent