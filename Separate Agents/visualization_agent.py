"""
Visualization Agent - Handles chart recommendations and configuration generation
Uses LLM for reasoning about best visualizations, then generates configs deterministically
"""

import requests
import json
from typing import List, Dict, Any, Optional

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

class VisualizationAgent:
    """
    Handles visualization-related tasks:
    - Recommending chart types
    - Generating Superset configurations
    - Suggesting visualization options
    """
    
    def recommend_visualizations(self, user_requirements: str, dataset_info: Dict, conversation_history: List[Dict]) -> List[Dict[str, Any]]:
        """
        PHASE 1: REASONING
        Use LLM to recommend appropriate visualizations based on user needs
        Now handles both specific AND vague requests intelligently
        """
        columns_summary = ', '.join([col['name'] for col in dataset_info['columns'][:15]])
        
        # Determine if user was specific or vague
        is_vague = len(user_requirements.split()) < 15 or user_requirements.count(" ") < 10
        
        if is_vague:
            # User was vague - give them diverse, useful options based on available data
            recommendation_prompt = f"""You are a visualization expert. The user wants visualizations but didn't provide many details.

WHAT THEY SAID:
"{user_requirements}"

AVAILABLE DATA:
- Table: {dataset_info['table_name']}
- Columns: {columns_summary}
- Date Column: {dataset_info.get('date_column', 'Unknown')}
- Row Count: {dataset_info['row_count']:,}

YOUR TASK: 
Since they didn't specify much, recommend 3 DIVERSE, USEFUL visualization options that would work well with this dataset. Cover different use cases.

STRATEGY:
1. **Option 1**: If there's a date column, suggest a TREND/TIME-BASED chart
2. **Option 2**: Suggest a COMPARISON/RANKING chart (top items)
3. **Option 3**: Suggest a BREAKDOWN/COMPOSITION chart (proportions)

Make them practical and business-focused. Use actual column names from the data.

RESPOND WITH JSON:
{{
  "recommendations": [
    {{
      "option_number": 1,
      "chart_type": "line",
      "title": "Trend Over Time (use actual date column name)",
      "description": "Shows how key metrics change over time. Great for spotting trends, seasonal patterns, and overall trajectory.",
      "columns_needed": ["date_column", "metric_column"],
      "best_for": "understanding trends and patterns over time"
    }},
    {{
      "option_number": 2,
      "chart_type": "bar",
      "title": "Top Performers Comparison (use actual columns)",
      "description": "Ranks and compares your best items. Perfect for identifying winners and focusing efforts.",
      "columns_needed": ["dimension_column", "metric_column"],
      "best_for": "identifying top performers and comparing items"
    }},
    {{
      "option_number": 3,
      "chart_type": "pie",
      "title": "Distribution Breakdown (use actual columns)",
      "description": "Shows how the total breaks down into parts. Useful for understanding composition and proportions.",
      "columns_needed": ["category_column", "metric_column"],
      "best_for": "understanding composition and market share"
    }}
  ]
}}

Be specific with actual column names. Respond with ONLY JSON."""

        else:
            # User was specific - tailor recommendations to their exact needs
            recommendation_prompt = f"""You are a visualization expert. Based on the user's specific requirements, recommend 3 tailored visualization options.

WHAT THE USER TOLD YOU:
"{user_requirements}"

AVAILABLE DATA:
- Table: {dataset_info['table_name']}
- Columns: {columns_summary}
- Date Column: {dataset_info.get('date_column', 'Unknown')}
- Row Count: {dataset_info['row_count']:,}

YOUR TASK: 
Recommend 3 SPECIFIC visualizations that directly address what they asked for. Each should provide a different perspective or insight.

RESPOND WITH JSON:
{{
  "recommendations": [
    {{
      "option_number": 1,
      "chart_type": "line|bar|pie|scatter",
      "title": "Specific title based on their exact requirements",
      "description": "Explain how this SPECIFICALLY addresses what they asked for",
      "columns_needed": ["col1", "col2"],
      "best_for": "the specific insight this provides for their stated goal"
    }},
    {{
      "option_number": 2,
      "chart_type": "line|bar|pie|scatter",
      "title": "Different angle on their requirements",
      "description": "How this offers different insights than Option 1",
      "columns_needed": ["col1", "col2"],
      "best_for": "alternative perspective on their needs"
    }},
    {{
      "option_number": 3,
      "chart_type": "line|bar|pie|scatter",
      "title": "Third unique perspective",
      "description": "Yet another way to address their goals",
      "columns_needed": ["col1", "col2"],
      "best_for": "third angle on their question"
    }}
  ]
}}

Respond with ONLY JSON."""

        try:
            payload = {
                "model": "mistralai/mistral-7b-instruct-v0.3",
                "messages": [{"role": "user", "content": recommendation_prompt}],
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False
            }
            
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            result = json.loads(content)
            return result.get("recommendations", [])
            
        except Exception as e:
            print(f"DEBUG [Viz Agent]: Error in recommendations: {e}")
            # Fallback recommendations
            return self._generate_fallback_recommendations(user_requirements, dataset_info)
    
    def _generate_fallback_recommendations(self, requirements: str, dataset_info: Dict) -> List[Dict]:
        """Fallback recommendations if LLM fails"""
        date_col = dataset_info.get('date_column', 'Date')
        metric_cols = [col['name'] for col in dataset_info['columns'] if 'dollar' in col['name'].lower() or 'sales' in col['name'].lower()]
        metric_col = metric_cols[0] if metric_cols else 'Value'
        
        dimension_cols = [col['name'] for col in dataset_info['columns'] if 'product' in col['name'].lower() or 'description' in col['name'].lower()]
        dimension_col = dimension_cols[0] if dimension_cols else dataset_info['columns'][0]['name']
        
        return [
            {
                "option_number": 1,
                "chart_type": "line",
                "title": "Trend Over Time",
                "description": "Shows how metrics change over time, revealing patterns and trends.",
                "columns_needed": [date_col, metric_col],
                "best_for": "identifying trends and patterns"
            },
            {
                "option_number": 2,
                "chart_type": "bar",
                "title": "Top 10 Comparison",
                "description": "Compares top items side-by-side for easy comparison.",
                "columns_needed": [dimension_col, metric_col],
                "best_for": "comparing performance across categories"
            },
            {
                "option_number": 3,
                "chart_type": "pie",
                "title": "Distribution Breakdown",
                "description": "Shows proportional breakdown of the whole.",
                "columns_needed": [dimension_col, metric_col],
                "best_for": "understanding composition and proportions"
            }
        ]
    
    def format_recommendations_for_user(self, recommendations: List[Dict]) -> str:
        """
        Format recommendations as friendly text for user
        """
        response = "Based on your needs, here are 3 visualization options:\n\n"
        
        for rec in recommendations:
            response += f"**Option {rec['option_number']}: {rec['chart_type'].title()} - {rec['title']}**\n"
            response += f"{rec['description']}\n"
            if rec.get('best_for'):
                response += f"*Best for: {rec['best_for']}*\n"
            response += "\n"
        
        response += "Which option sounds most helpful? Just say 'Option 1', 'Option 2', or 'Option 3'!"
        
        return response
    
    def generate_superset_config(self, visualization: Dict, sql_query: str, data_sample: List[Dict]) -> str:
        """
        PHASE 2: EXECUTION
        Generate Superset YAML configuration deterministically
        No LLM needed - just template filling
        """
        chart_type = visualization.get('chart_type', 'line')
        title = visualization.get('title', 'Visualization')
        
        # Extract columns from data sample
        if data_sample:
            columns = list(data_sample[0].keys())
            x_axis = columns[0] if columns else 'x'
            y_axis = columns[1] if len(columns) > 1 else 'y'
        else:
            x_axis = 'x'
            y_axis = 'y'
        
        config = f"""title: "{title}"
description: "{visualization.get('description', 'Custom visualization')}"
chart_type: "{chart_type}"

data_source:
  query: |
{self._indent_sql(sql_query, 4)}

visualization:
  type: "{chart_type}"
  x_axis: "{x_axis}"
  y_axis: "{y_axis}"

dimensions:
  - "{x_axis}"

metrics:
  - name: "{y_axis}"
    expression: "{y_axis}"
    format: "number"

grouping:
  group_by: ["{x_axis}"]
  sort_by: "{y_axis}"
  sort_order: "DESC"
  limit: 50
"""
        
        return config
    
    def _indent_sql(self, sql: str, spaces: int) -> str:
        """Indent SQL query for YAML formatting"""
        indent = ' ' * spaces
        lines = sql.split('\n')
        return '\n'.join([indent + line for line in lines])
    
    def ask_dashboard_questions(self, user_prompt: str, dataset_info: Dict) -> str:
        """
        Ask clarifying questions for dashboard building
        PRIVACY SAFE: Only uses column names and metadata, no actual data
        """
        # Extract column names only - no actual values
        column_names = ', '.join([col['name'] for col in dataset_info['columns'][:10]])
        
        instruction = f"""You are Aevah, a helpful data analyst. The user wants to build visualizations.

DATASET STRUCTURE (NO ACTUAL DATA SHOWN FOR PRIVACY):
- Table: {dataset_info['table_name']}
- Columns Available: {column_names}
- Total Records: {dataset_info['row_count']:,}
- Date Column: {dataset_info.get('date_column', 'Unknown')}

USER SAID: "{user_prompt}"

YOUR TASK:
Ask 2-3 SHORT, conversational questions to understand what they need. Focus on:

1. **What metric/data** do they want to visualize? (Look at column names for ideas)
2. **What time period** are they interested in?
3. **What's the purpose?** (trends, comparisons, distribution)

Be warm and conversational. Keep it brief!

IMPORTANT: You can see the STRUCTURE of their data (column names), but NOT the actual values - this protects their privacy.

GOOD EXAMPLE:
"I'd love to help you create some great visualizations! 

To make sure I build exactly what you need, I have a few quick questions:

1. What specific data or metric are you most interested in seeing? (I can see you have columns like {column_names[:50]}...)
2. Is there a particular time period you want to focus on?
3. What are you hoping to learn or discover?

Let me know and I'll suggest some perfect options!"

Generate a similar friendly response."""

        try:
            payload = {
                "model": "mistralai/mistral-7b-instruct-v0.3",
                "messages": [{"role": "user", "content": instruction}],
                "temperature": 0.7,
                "max_tokens": 400,
                "stream": False
            }
            
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            # Fallback questions (privacy safe - no data)
            return f"""I'd love to help you build some great visualizations! 

To create the most useful charts for you, I have a few quick questions:

1. **What data are you most interested in?** (I can see you have: {column_names[:80]}...)
2. **Any specific time period?** (e.g., last year, Q3 2023, recent months)
3. **What's your goal?** (spot trends, compare items, see top performers)

Just answer in your own words!"""
    
    def ask_simple_followup(self, user_answers: List[str], dataset_info: Dict) -> Optional[str]:
        """
        Ask ONE simple follow-up if user was really vague
        Much more flexible - we don't need all details
        """
        previous_answers = " ".join(user_answers).lower()
        
        # If user said almost nothing useful, ask one gentle question
        if len(previous_answers.split()) < 5:
            return "Could you tell me a bit more about what you'd like to see? For example, are you interested in trends, comparisons, or something else?"
        
        # Otherwise, we have enough to work with
        return None
    
    def ask_followup_question(self, user_answers: List[str], dataset_info: Dict, 
                              has_metric: bool, has_timeframe: bool, has_purpose: bool) -> Optional[str]:
        """
        Ask targeted follow-up questions based on what information is still missing
        """
        # Determine what's missing
        missing = []
        if not has_metric:
            missing.append("metric/data")
        if not has_timeframe:
            missing.append("time period")
        if not has_purpose:
            missing.append("purpose/goal")
        
        if not missing:
            return None  # We have everything we need
        
        # Build contextual follow-up
        previous_answers = " ".join(user_answers)
        
        instruction = f"""You are Aevah. The user is answering questions about visualizations they want.

DATASET:
- Available Columns: {', '.join([col['name'] for col in dataset_info['columns'][:10]])}
- Date Column: {dataset_info.get('date_column', 'Unknown')}

WHAT USER HAS SAID SO FAR:
"{previous_answers}"

WHAT WE STILL NEED TO KNOW:
{', '.join(missing)}

YOUR TASK:
Ask ONE brief, conversational follow-up question to get the missing information.

EXAMPLES:
- If missing metric: "Great! And which specific metric would you like to focus on? (e.g., total sales, units sold, performance over time)"
- If missing timeframe: "Perfect! Is there a specific time period you'd like to analyze? (e.g., 2023, last quarter, recent months)"
- If missing purpose: "Sounds good! What are you hoping to discover? (trends, comparisons, top performers, etc.)"

Generate ONE brief follow-up question."""

        try:
            payload = {
                "model": "mistralai/mistral-7b-instruct-v0.3",
                "messages": [{"role": "user", "content": instruction}],
                "temperature": 0.7,
                "max_tokens": 200,
                "stream": False
            }
            
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            # Fallback based on what's missing
            if not has_metric:
                return "Great! Which specific metric would you like to focus on? (e.g., sales, units, revenue, performance)"
            elif not has_timeframe:
                return "Perfect! Is there a particular time period you'd like to analyze? (e.g., 2023, last quarter, recent data)"
            elif not has_purpose:
                return "Sounds good! What are you hoping to discover from this visualization? (trends, comparisons, top items, etc.)"
            return None


# Singleton instance
_visualization_agent = None

def get_visualization_agent() -> VisualizationAgent:
    """Get singleton visualization agent instance"""
    global _visualization_agent
    if _visualization_agent is None:
        _visualization_agent = VisualizationAgent()
    return _visualization_agent