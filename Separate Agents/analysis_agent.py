"""
Analysis Agent - Handles data insights, explanations, and forecasting
Focuses on interpreting data and providing business intelligence
"""

import requests
import json
from typing import List, Dict, Any

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

class AnalysisAgent:
    """
    Handles analytical tasks:
    - Analyzing visualization data
    - Providing insights
    - Forecasting
    - Generating executive summaries
    """
    
    def analyze_visualization(self, user_question: str, chart_data: Dict, dataset_info: Dict) -> str:
        """
        Analyze visualization and provide insights based on user question
        PRIVACY SAFE: Analyzes AGGREGATED results, not raw customer data
        """
        data = chart_data.get('preview_data', [])
        chart_type = chart_data.get('chart_type', 'chart')
        
        if not data:
            return "I don't have enough data to analyze this visualization."
        
        # PRIVACY: Data shown in charts is already AGGREGATED (sums, counts, averages)
        # Not raw customer records, so safe to analyze
        columns = list(data[0].keys())
        row_count = len(data)
        
        # Create summary statistics WITHOUT exposing individual records
        summary_stats = self._create_summary_stats(data, columns)
        
        analysis_prompt = f"""You are Aevah, a data analyst. The user is looking at a {chart_type} and asked: "{user_question}"

VISUALIZATION SUMMARY (AGGREGATED DATA - NO INDIVIDUAL RECORDS):
- Chart Type: {chart_type}
- Number of Data Points: {row_count}
- Dimensions: {', '.join(columns)}

SUMMARY STATISTICS:
{json.dumps(summary_stats, indent=2)}

TOP 3 VALUES:
{json.dumps(data[:3], indent=2)}

BOTTOM 3 VALUES (if applicable):
{json.dumps(data[-3:] if len(data) > 3 else [], indent=2)}

YOUR TASK:
Analyze this AGGREGATED data and provide specific insights. Focus on:
1. Key patterns or trends
2. Notable findings (highest/lowest, outliers)
3. Business implications
4. Actionable observations

BE SPECIFIC - reference actual values from the summary.
BE CONVERSATIONAL - talk like a helpful colleague.
BE CONCISE - 3-4 sentences unless asked for detail.

IMPORTANT: This is aggregated/summarized data (totals, averages, counts), not individual customer records.

Example:
"Looking at this chart, [Item X] is clearly the top performer with [aggregate_value] - that's [%] higher than [Item Y]. I also notice [specific trend or pattern in the aggregates]. This suggests [business insight based on aggregated data]."

Respond naturally."""

        try:
            payload = {
                "model": "mistralai/mistral-7b-instruct-v0.3",
                "messages": [{"role": "user", "content": analysis_prompt}],
                "temperature": 0.7,
                "max_tokens": 500,
                "stream": False
            }
            
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=45)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            # Fallback analysis
            return self._generate_fallback_analysis(data, chart_type)
    
    def _create_summary_stats(self, data: List[Dict], columns: List[str]) -> Dict:
        """Create summary statistics from aggregated data (privacy safe)"""
        stats = {}
        
        for col in columns:
            try:
                values = [row.get(col) for row in data if row.get(col) is not None]
                
                # Check if numeric
                numeric_values = []
                for v in values:
                    try:
                        numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        pass
                
                if numeric_values:
                    stats[col] = {
                        "type": "numeric",
                        "count": len(numeric_values),
                        "max": max(numeric_values),
                        "min": min(numeric_values),
                        "avg": sum(numeric_values) / len(numeric_values) if numeric_values else 0
                    }
                else:
                    # Categorical
                    stats[col] = {
                        "type": "categorical",
                        "count": len(values),
                        "unique_count": len(set(str(v) for v in values))
                    }
            except Exception as e:
                continue
        
        return stats
    
    def _generate_fallback_analysis(self, data: List[Dict], chart_type: str) -> str:
        """Generate basic analysis if LLM fails"""
        if not data:
            return "No data available to analyze."
        
        columns = list(data[0].keys())
        row_count = len(data)
        
        analysis = f"Looking at this {chart_type}, I can see {row_count} data points across {len(columns)} dimensions: {', '.join(columns)}. "
        
        # Find top value
        if len(columns) >= 2:
            value_col = columns[1]
            top_item = max(data, key=lambda x: float(x.get(value_col, 0)) if isinstance(x.get(value_col), (int, float)) or str(x.get(value_col, '0')).replace('.','').isdigit() else 0)
            analysis += f"The highest value is {top_item.get(columns[0], 'Unknown')} with {top_item.get(value_col, 0)}. "
        
        analysis += "What specific aspect would you like me to explain in more detail?"
        
        return analysis
    
    def generate_forecast(self, historical_data: List[Dict], metric_name: str, periods: int, forecast_unit: str) -> Dict[str, Any]:
        """
        Generate simple forecast using moving average with trend
        This is deterministic - no LLM needed
        """
        if not historical_data or len(historical_data) < 2:
            return {
                "forecast_values": [],
                "method": "insufficient_data",
                "confidence": "low",
                "explanation": "Not enough historical data to generate a forecast."
            }
        
        # Extract values
        values = []
        for row in historical_data:
            val = row.get(metric_name)
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue
        
        if len(values) < 2:
            return {
                "forecast_values": [],
                "method": "insufficient_data",
                "confidence": "low",
                "explanation": "Not enough valid data points to forecast."
            }
        
        # Calculate moving average and trend
        window = min(3, len(values))
        recent_avg = sum(values[-window:]) / window
        
        if len(values) >= 3:
            recent_trend = (values[-1] - values[-3]) / 3
        else:
            recent_trend = values[-1] - values[-2]
        
        # Generate forecasts
        forecasts = []
        last_value = values[-1]
        
        for i in range(1, periods + 1):
            trend_forecast = last_value + (recent_trend * i)
            forecast_value = (trend_forecast * 0.7) + (recent_avg * 0.3)
            forecasts.append(round(forecast_value, 2))
        
        # Calculate confidence
        if len(values) >= 4:
            variance = sum((v - recent_avg) ** 2 for v in values[-4:]) / 4
            std_dev = variance ** 0.5
            cv = (std_dev / recent_avg) * 100 if recent_avg != 0 else 100
            confidence = "high" if cv < 10 else "medium" if cv < 25 else "low"
        else:
            confidence = "medium"
        
        return {
            "forecast_values": forecasts,
            "method": "moving_average_with_trend",
            "confidence": confidence,
            "recent_average": round(recent_avg, 2),
            "trend": round(recent_trend, 2),
            "historical_last_value": round(last_value, 2),
            "explanation": f"Based on recent trends, projecting {periods} {forecast_unit}(s) ahead."
        }
    
    def format_forecast_response(self, forecast_result: Dict, metric_name: str, periods: int, forecast_unit: str) -> str:
        """Format forecast results for display"""
        if forecast_result["method"] == "insufficient_data":
            return "I don't have enough historical data to generate a reliable forecast."
        
        forecast_values = forecast_result["forecast_values"]
        confidence = forecast_result["confidence"]
        
        response = f"📊 **Forecast Analysis for {metric_name}**\n\n"
        response += "**Historical Context:**\n"
        response += f"- Recent average: {forecast_result.get('recent_average', 'N/A'):,.2f}\n"
        response += f"- Last value: {forecast_result.get('historical_last_value', 'N/A'):,.2f}\n"
        response += f"- Trend: {'+' if forecast_result.get('trend', 0) > 0 else ''}{forecast_result.get('trend', 'N/A'):,.2f} per period\n\n"
        
        response += f"**Forecast for next {periods} {forecast_unit}(s):**\n"
        for i, val in enumerate(forecast_values, 1):
            response += f"- Period {i}: {val:,.2f}\n"
        
        response += f"\n**Confidence Level:** {confidence.upper()}\n"
        response += f"**Method:** {forecast_result['method'].replace('_', ' ').title()}\n\n"
        
        if confidence == "low":
            response += "⚠️ *Note: Forecast has low confidence due to data variability. Use with caution.*"
        elif confidence == "high":
            response += "✅ *Forecast has high confidence based on consistent historical trends.*"
        
        return response
    
    def generate_executive_summary(self, dataset_info: Dict) -> str:
        """
        Generate executive summary of dataset
        """
        table_name = dataset_info['table_name']
        row_count = dataset_info['row_count']
        columns = [col['name'] for col in dataset_info['columns'][:10]]
        sample_data = json.dumps(dataset_info['sample_data'][0]) if dataset_info['sample_data'] else "No sample"
        
        summary_prompt = f"""Generate an executive summary for this dataset.

DATASET:
- Name: {table_name}
- Records: {row_count:,}
- Columns: {', '.join(columns)}
- Sample: {sample_data}

Format:
**EXECUTIVE SUMMARY: {table_name}**

**1. Overview**
[2-3 sentences about what this data represents]

**2. Key Metrics Available**
- [Metric 1 description]
- [Metric 2 description]

**3. Analytical Opportunities**
1. [Analysis type 1]
2. [Analysis type 2]

Be business-focused and actionable."""

        try:
            payload = {
                "model": "mistralai/mistral-7b-instruct-v0.3",
                "messages": [{"role": "user", "content": summary_prompt}],
                "temperature": 0.6,
                "max_tokens": 800,
                "stream": False
            }
            
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            # Fallback summary
            return self._generate_fallback_summary(dataset_info)
    
    def _generate_fallback_summary(self, dataset_info: Dict) -> str:
        """Generate basic summary if LLM fails"""
        return f"""**EXECUTIVE SUMMARY: {dataset_info['table_name']}**

**1. Overview**
This dataset contains {dataset_info['row_count']:,} records with {len(dataset_info['columns'])} data columns. It includes information about {', '.join([col['name'] for col in dataset_info['columns'][:5]])}.

**2. Key Metrics Available**
- Data spans multiple dimensions for comprehensive analysis
- Includes temporal and categorical data for trend analysis
- Contains metrics suitable for performance tracking

**3. Analytical Opportunities**
1. Trend analysis over time periods
2. Comparative analysis across categories
3. Performance benchmarking and forecasting

You can ask me specific questions about any aspect of this data!"""


# Singleton instance
_analysis_agent = None

def get_analysis_agent() -> AnalysisAgent:
    """Get singleton analysis agent instance"""
    global _analysis_agent
    if _analysis_agent is None:
        _analysis_agent = AnalysisAgent()
    return _analysis_agent