# sql_agent.py - FIXED VERSION
"""
Robust SQL agent with proper WHERE clause building.
Handles complex queries, proper filter parsing, and LLM fallback.
"""

import sqlite3
import re
import requests
import json
from typing import List, Dict, Any, Optional, Tuple

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
DB_PATH = "app_data.db"


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


class SQLAgent:
    def __init__(self):
        self.dataset_cache: Dict[str, Dict[str, Any]] = {}

    def get_dataset_info(self, table_name: str) -> Dict[str, Any]:
        """Get metadata with sample values"""
        if table_name in self.dataset_cache:
            return self.dataset_cache[table_name]

        conn = get_conn()
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
            row_count = cursor.fetchone()[0]
        except:
            row_count = 0

        cursor.execute(f"PRAGMA table_info([{table_name}])")
        columns = [{"name": c[1], "type": c[2]} for c in cursor.fetchall()]
        col_names = [c['name'] for c in columns]
        
        sample_values = {}
        unique_counts = {}
        for col in col_names[:20]:
            try:
                cursor.execute(f"SELECT DISTINCT [{col}] FROM [{table_name}] WHERE [{col}] IS NOT NULL LIMIT 5")
                samples = [row[0] for row in cursor.fetchall()]
                sample_values[col] = samples
                cursor.execute(f"SELECT COUNT(DISTINCT [{col}]) FROM [{table_name}]")
                unique_counts[col] = cursor.fetchone()[0]
            except:
                sample_values[col] = []
                unique_counts[col] = 0
        
        column_roles = self._analyze_columns(columns, sample_values, unique_counts)
        conn.close()

        info = {
            "table_name": table_name,
            "row_count": row_count,
            "columns": columns,
            "column_names": col_names,
            "sample_values": sample_values,
            "unique_counts": unique_counts,
            "column_roles": column_roles
        }
        self.dataset_cache[table_name] = info
        return info

    def _analyze_columns(self, columns: List[Dict], sample_values: Dict, unique_counts: Dict) -> Dict:
        """Smart column role detection"""
        roles = {
            "product_col": None,
            "metric_col": None,
            "date_col": None,
            "all_products": [],
            "all_metrics": [],
            "all_dates": [],
            "all_filters": []
        }
        
        for col in columns:
            name = col['name']
            nl = name.lower()
            samples = sample_values.get(name, [])
            
            is_numeric = False
            if samples:
                try:
                    float(str(samples[0]).replace(',', '').replace('$', ''))
                    is_numeric = True
                except:
                    pass
            
            # Product columns
            if 'description' in nl or nl == 'desc':
                roles["product_col"] = name
                roles["all_products"].insert(0, name)
            elif 'upc' in nl or 'sku' in nl:
                roles["all_products"].append(name)
                if not roles["product_col"]:
                    roles["product_col"] = name
            elif 'flavor' in nl or 'item' in nl:
                roles["all_products"].append(name)
            
            # Metric columns
            if nl == 'dollars' or nl == 'units':
                if not roles["metric_col"]:
                    roles["metric_col"] = name
                roles["all_metrics"].insert(0, name)
            elif 'dollar' in nl and 'spm' not in nl and 'avg' not in nl:
                roles["all_metrics"].append(name)
            elif 'unit' in nl and 'spm' not in nl and 'avg' not in nl:
                roles["all_metrics"].append(name)
            elif is_numeric and 'spm' not in nl and '%' not in name:
                roles["all_metrics"].append(name)
            
            # Date columns
            if 'date' in nl or 'period' in nl:
                roles["all_dates"].append(name)
                if 'end' in nl:
                    roles["date_col"] = name
                elif not roles["date_col"]:
                    roles["date_col"] = name
            
            # Filter columns
            if is_numeric and unique_counts.get(name, 0) < 100:
                roles["all_filters"].append({"name": name, "type": "numeric"})
        
        if not roles["product_col"] and roles["all_products"]:
            roles["product_col"] = roles["all_products"][0]
        if not roles["metric_col"] and roles["all_metrics"]:
            roles["metric_col"] = roles["all_metrics"][0]
        
        return roles

    def _parse_user_requirements(self, prompt: str, dataset_info: Dict) -> Dict:
        """Parse user prompt into structured requirements"""
        prompt_lower = prompt.lower()
        roles = dataset_info.get('column_roles', {})
        col_names = dataset_info.get('column_names', [])
        
        requirements = {
            "query_type": "top_n",
            "product_col": roles.get("product_col", "Description"),
            "metric_col": roles.get("metric_col", "Dollars"),
            "date_col": roles.get("date_col"),
            "limit": 50,
            "order": "DESC",
            "filters": [],
            "year_filter": None,
            "group_by_time": False,
            "needs_llm": False,  # Flag for complex queries
            "complex_query": None  # Store detected complex query type
        }
        
        # DETECT COMPLEX QUERIES that need LLM
        complex_patterns = {
            "cannibalization": ["cannibaliz", "cannibalize", "cannibal"],
            "price_elasticity": ["price elasticity", "elasticity"],
            "correlation": ["correlation", "correlate"],
            "market_share": ["market share", "share of"],
            "basket_analysis": ["basket", "bought together", "frequently purchased"],
        }
        
        for query_type, patterns in complex_patterns.items():
            if any(p in prompt_lower for p in patterns):
                requirements["needs_llm"] = True
                requirements["complex_query"] = query_type
                break
        
        # Detect query type
        if any(w in prompt_lower for w in ["trend", "over time", "by week", "by month", "timeline"]):
            requirements["query_type"] = "trend"
            requirements["group_by_time"] = True
        elif any(w in prompt_lower for w in ["yoy", "year over year", "declining", "year-over-year"]):
            requirements["query_type"] = "yoy"
        elif any(w in prompt_lower for w in ["new product", "introduced", "first time", "launched"]):
            requirements["query_type"] = "new_products"
        elif any(w in prompt_lower for w in ["bottom", "lowest", "worst", "least"]):
            requirements["order"] = "ASC"
        
        # Detect limit
        limit_match = re.search(r'top\s*(\d+)', prompt_lower)
        if limit_match:
            requirements["limit"] = int(limit_match.group(1))
        
        # Detect year filter
        year_match = re.search(r'\b(20\d{2})\b', prompt)
        if year_match:
            requirements["year_filter"] = year_match.group(1)
        
        # Detect metric preference
        if "dollar" in prompt_lower or "sales" in prompt_lower or "revenue" in prompt_lower:
            for col in col_names:
                if col.lower() == 'dollars':
                    requirements["metric_col"] = col
                    break
        elif "unit" in prompt_lower or "volume" in prompt_lower:
            for col in col_names:
                if col.lower() == 'units':
                    requirements["metric_col"] = col
                    break
        
        # Detect numeric filters - FIXED to avoid duplicate detection
        detected_filters = set()  # Track what we've already added
        
        for col in col_names:
            col_lower = col.lower()
            col_normalized = col_lower.replace(' ', '').replace('_', '')
            
            # Check if this column is mentioned
            if col_lower in prompt_lower or col_normalized in prompt_lower.replace(' ', ''):
                # Skip if already detected for this column
                if col in detected_filters:
                    continue
                
                # Look for numeric conditions
                conditions = [
                    (r'(\d+)\s*or\s*more', '>='),
                    (r'(\d+)\s*or\s*greater', '>='),
                    (r'>=\s*(\d+)', '>='),
                    (r'at\s*least\s*(\d+)', '>='),
                    (r'more\s*than\s*(\d+)', '>'),
                    (r'greater\s*than\s*(\d+)', '>'),
                    (r'less\s*than\s*(\d+)', '<'),
                    (r'<=\s*(\d+)', '<='),
                ]
                
                for regex, op in conditions:
                    match = re.search(regex, prompt_lower)
                    if match:
                        value = int(match.group(1))
                        # Only add if this is a filterable column (not the main metric)
                        if col != requirements["metric_col"]:
                            requirements["filters"].append({
                                "column": col,
                                "operator": op,
                                "value": value
                            })
                            detected_filters.add(col)
                        break
        
        return requirements

    def _build_sql_from_requirements(self, req: Dict, table_name: str, dataset_info: Dict) -> str:
        """Build SQL programmatically - FIXED WHERE clause"""
        product_col = req["product_col"]
        metric_col = req["metric_col"]
        date_col = req.get("date_col")
        
        # Collect all WHERE conditions
        where_conditions = []
        
        # Year filter
        if req.get("year_filter") and date_col:
            where_conditions.append(f"[{date_col}] LIKE '%{req['year_filter']}%'")
        
        # Numeric filters
        for f in req.get("filters", []):
            where_conditions.append(f"[{f['column']}] {f['operator']} {f['value']}")
        
        # Product not null
        where_conditions.append(f"[{product_col}] IS NOT NULL")
        
        # Build WHERE clause
        where_sql = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        
        # Build query based on type
        if req["query_type"] == "trend":
            sql = f"""SELECT [{date_col}] AS period, SUM([{metric_col}]) AS total
FROM [{table_name}]
{where_sql}
GROUP BY [{date_col}]
ORDER BY [{date_col}]
LIMIT 100"""
        
        elif req["query_type"] == "yoy":
            sql = f"""WITH yearly AS (
    SELECT [{product_col}] AS product,
           CASE 
               WHEN [{date_col}] LIKE '%2024%' THEN '2024'
               WHEN [{date_col}] LIKE '%2023%' THEN '2023'
               WHEN [{date_col}] LIKE '%2022%' THEN '2022'
               ELSE 'other'
           END AS year,
           SUM([{metric_col}]) AS total
    FROM [{table_name}]
    WHERE [{product_col}] IS NOT NULL
    GROUP BY [{product_col}], year
)
SELECT 
    cy.product,
    ROUND(cy.total, 2) AS current_year,
    ROUND(py.total, 2) AS prior_year,
    ROUND(cy.total - COALESCE(py.total, 0), 2) AS change
FROM yearly cy
LEFT JOIN yearly py ON cy.product = py.product AND CAST(cy.year AS INT) = CAST(py.year AS INT) + 1
WHERE cy.year = (SELECT MAX(year) FROM yearly WHERE year != 'other')
  AND (cy.total - COALESCE(py.total, 0)) < 0
ORDER BY change ASC
LIMIT {req['limit']}"""
        
        elif req["query_type"] == "new_products":
            year = req.get("year_filter", "2023")
            sql = f"""WITH first_seen AS (
    SELECT [{product_col}] AS product, MIN([{date_col}]) AS first_date
    FROM [{table_name}]
    WHERE [{product_col}] IS NOT NULL
    GROUP BY [{product_col}]
)
SELECT product, first_date
FROM first_seen
WHERE first_date LIKE '%{year}%'
ORDER BY first_date
LIMIT {req['limit']}"""
        
        else:  # top_n (default)
            sql = f"""SELECT [{product_col}] AS product, SUM([{metric_col}]) AS total
FROM [{table_name}]
{where_sql}
GROUP BY [{product_col}]
ORDER BY total {req['order']}
LIMIT {req['limit']}"""
        
        return sql.strip()

    def _validate_sql(self, sql: str, table_name: str) -> Tuple[bool, str]:
        """Validate SQL syntax"""
        try:
            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            conn.close()
            return True, ""
        except Exception as e:
            return False, str(e)

    def execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL and return results"""
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        cols = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()
        return [dict(zip(cols, r)) for r in rows]

    def _try_llm_for_complex_query(self, prompt: str, table_name: str, dataset_info: Dict, query_type: str) -> Optional[str]:
        """Use LLM for complex analytical queries"""
        roles = dataset_info.get('column_roles', {})
        product = roles.get('product_col', 'Description')
        metric = roles.get('metric_col', 'Dollars')
        date_col = roles.get('date_col', 'Time Period End Date')
        
        # Provide context based on query type
        query_guidance = {
            "cannibalization": f"""For cannibalization analysis:
- Compare products that grew vs declined in the same time period
- Look for negative correlation patterns
- Simple approach: show products with declining sales while category grew""",
            "price_elasticity": f"""For price elasticity:
- Need price and quantity data
- Calculate % change in quantity / % change in price
- If no price column, explain this to user""",
            "correlation": """For correlation:
- Return pairs of numeric values for client-side calculation
- Or compute simple comparison metrics""",
        }
        
        guidance = query_guidance.get(query_type, "Generate appropriate analytical SQL")
        
        llm_prompt = f"""Generate SQLite query for: {prompt}

Table: [{table_name}]
Key columns:
- Product: [{product}]
- Metric: [{metric}]  
- Date: [{date_col}]
All columns: {', '.join(dataset_info['column_names'][:15])}

{guidance}

Rules:
- Wrap ALL columns in [brackets]
- Keep it simple - prefer basic aggregations
- If the analysis isn't possible with available data, generate a simple top N query instead

Return ONLY the SQL query:"""

        try:
            payload = {
                "model": "mistralai/mistral-7b-instruct-v0.3",
                "messages": [{"role": "user", "content": llm_prompt}],
                "temperature": 0.1,
                "max_tokens": 500,
                "stream": False
            }
            
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
            response.raise_for_status()
            sql = response.json()["choices"][0]["message"]["content"].strip()
            sql = sql.replace("```sql", "").replace("```", "").strip()
            
            # Validate
            is_valid, _ = self._validate_sql(sql, table_name)
            if is_valid:
                return sql
        except Exception as e:
            print(f"DEBUG [SQL Agent]: LLM failed: {e}")
        
        return None

    def handle_analytical_query(self, user_prompt: str, table_name: str, context: Dict = None) -> Dict[str, Any]:
        """Main entry point"""
        print(f"DEBUG [SQL Agent]: Processing: {user_prompt}")
        
        dataset_info = self.get_dataset_info(table_name)
        
        # Apply user feedback if present
        if context and context.get('user_feedback'):
            user_prompt = f"{user_prompt}. {context['user_feedback']}"
        
        # Parse requirements
        requirements = self._parse_user_requirements(user_prompt, dataset_info)
        print(f"DEBUG [SQL Agent]: Parsed: {requirements}")
        
        # Check if this needs LLM for complex analysis
        if requirements.get("needs_llm") and requirements.get("complex_query"):
            print(f"DEBUG [SQL Agent]: Complex query detected: {requirements['complex_query']}")
            llm_sql = self._try_llm_for_complex_query(
                user_prompt, table_name, dataset_info, requirements["complex_query"]
            )
            if llm_sql:
                try:
                    data = self.execute_query(llm_sql)
                    return self._build_response(data, llm_sql, requirements, dataset_info)
                except Exception as e:
                    print(f"DEBUG [SQL Agent]: LLM SQL failed: {e}")
        
        # Build SQL programmatically
        sql_query = self._build_sql_from_requirements(requirements, table_name, dataset_info)
        print(f"DEBUG [SQL Agent]: Built SQL:\n{sql_query}")
        
        # Validate
        is_valid, error = self._validate_sql(sql_query, table_name)
        if not is_valid:
            print(f"DEBUG [SQL Agent]: Validation failed: {error}")
            sql_query = self._simple_fallback(dataset_info)
        
        # Execute
        try:
            data = self.execute_query(sql_query)
        except Exception as e:
            print(f"DEBUG [SQL Agent]: Execution failed: {e}")
            sql_query = self._simple_fallback(dataset_info)
            data = self.execute_query(sql_query)
        
        return self._build_response(data, sql_query, requirements, dataset_info)

    def _build_response(self, data: List[Dict], sql_query: str, requirements: Dict, dataset_info: Dict) -> Dict:
        """Build standard response"""
        chart_type = "bar"
        if requirements.get("group_by_time") or requirements.get("query_type") == "trend":
            chart_type = "line"
        
        x_axis = "product"
        y_axis = "total"
        if data:
            cols = list(data[0].keys())
            x_axis = cols[0] if cols else "x"
            y_axis = cols[1] if len(cols) > 1 else "y"
        
        return {
            "data": data,
            "sql_query": sql_query,
            "explanation": f"Query: {requirements.get('query_type', 'top_n')}, Filters: {requirements.get('filters', [])}",
            "x_axis": x_axis,
            "y_axis": y_axis,
            "chart_type": chart_type,
            "row_count": len(data),
            "success": True
        }

    def _simple_fallback(self, dataset_info: Dict) -> str:
        """Simple fallback query"""
        roles = dataset_info.get('column_roles', {})
        product = roles.get('product_col', 'Description')
        metric = roles.get('metric_col', 'Dollars')
        table = dataset_info['table_name']
        
        return f"""SELECT [{product}] AS product, SUM([{metric}]) AS total
FROM [{table}]
WHERE [{product}] IS NOT NULL
GROUP BY [{product}]
ORDER BY total DESC
LIMIT 50"""

    def handle_visualization_query(self, viz_spec: Dict, table_name: str) -> Dict[str, Any]:
        """Handle visualization queries"""
        title = viz_spec.get('title', '')
        return self.handle_analytical_query(f"Show data for {title}", table_name, None)

    def clear_cache(self, table_name: str = None):
        if table_name:
            self.dataset_cache.pop(table_name, None)
        else:
            self.dataset_cache.clear()


_sql_agent: Optional[SQLAgent] = None

def get_sql_agent() -> SQLAgent:
    global _sql_agent
    if _sql_agent is None:
        _sql_agent = SQLAgent()
    return _sql_agent