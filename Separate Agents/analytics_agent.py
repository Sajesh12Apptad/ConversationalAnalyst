# analytics_agent.py
"""
Analytics Agent - Handles complex statistical computations
Price Elasticity, Cannibalization, Correlation, etc.
"""

import sqlite3
from typing import List, Dict, Any, Optional
import statistics

DB_PATH = "app_data.db"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


class AnalyticsAgent:
    """Handles complex analytical computations that can't be done in SQL alone"""
    
    def __init__(self):
        self.supported_analyses = {
            "price_elasticity": self.calculate_price_elasticity,
            "cannibalization": self.analyze_cannibalization,
            "correlation": self.calculate_correlation,
            "yoy_growth": self.calculate_yoy_growth,
        }
    
    def can_handle(self, query_type: str) -> bool:
        """Check if this agent can handle the query type"""
        return query_type in self.supported_analyses
    
    def execute(self, query_type: str, table_name: str, params: Dict) -> Dict[str, Any]:
        """Execute the appropriate analysis"""
        if query_type in self.supported_analyses:
            return self.supported_analyses[query_type](table_name, params)
        return {"error": f"Unknown analysis type: {query_type}"}
    
    def _get_column_names(self, table_name: str) -> Dict[str, str]:
        """Detect relevant column names from the table"""
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info([{table_name}])")
        columns = [c[1] for c in cursor.fetchall()]
        conn.close()
        
        result = {
            "product": None,
            "units": None,
            "price": None,
            "date": None
        }
        
        for col in columns:
            cl = col.lower()
            if 'description' in cl or cl == 'desc':
                result["product"] = col
            elif cl == 'units' or 'unit' in cl and 'spm' not in cl:
                result["units"] = col
            elif 'base dollar' in cl or cl == 'base dollars':
                result["price"] = col
            elif 'dollar' in cl and 'base' not in cl and 'spm' not in cl:
                if not result["price"]:
                    result["price"] = col
            elif 'date' in cl or 'period' in cl:
                if 'end' in cl:
                    result["date"] = col
                elif not result["date"]:
                    result["date"] = col
        
        # Fallbacks
        if not result["product"]:
            result["product"] = "Description"
        if not result["units"]:
            result["units"] = "Units"
        if not result["price"]:
            result["price"] = "Base Dollars"
        if not result["date"]:
            result["date"] = "Time Period End Date"
        
        return result
    
    def calculate_price_elasticity(self, table_name: str, params: Dict) -> Dict[str, Any]:
        """
        Calculate Price Elasticity of Demand for each product
        PED = (% Change in Quantity Demanded) / (% Change in Price)
        """
        year = params.get("year", "2023")
        limit = params.get("limit", 20)
        
        cols = self._get_column_names(table_name)
        product_col = cols["product"]
        units_col = cols["units"]
        price_col = cols["price"]
        date_col = cols["date"]
        
        # Get data for the year
        conn = get_conn()
        cursor = conn.cursor()
        
        sql = f"""
        SELECT [{product_col}] AS product, 
               [{units_col}] AS units, 
               [{price_col}] AS price,
               [{date_col}] AS period
        FROM [{table_name}]
        WHERE [{date_col}] LIKE '%{year}%'
          AND [{product_col}] IS NOT NULL
          AND [{units_col}] IS NOT NULL
          AND [{price_col}] IS NOT NULL
          AND [{units_col}] > 0
          AND [{price_col}] > 0
        ORDER BY [{product_col}], [{date_col}]
        """
        
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                "success": False,
                "error": f"No data found for year {year}",
                "data": [],
                "sql_query": sql
            }
        
        # Group by product
        products = {}
        for row in rows:
            product, units, price, period = row
            if product not in products:
                products[product] = {"units": [], "prices": []}
            try:
                products[product]["units"].append(float(units))
                products[product]["prices"].append(float(price))
            except (ValueError, TypeError):
                continue
        
        # Calculate elasticity for each product
        results = []
        for product, data in products.items():
            if len(data["units"]) < 2 or len(data["prices"]) < 2:
                continue
            
            # Average values
            avg_units = statistics.mean(data["units"])
            avg_price = statistics.mean(data["prices"])
            
            if avg_units == 0 or avg_price == 0:
                continue
            
            # Calculate % changes from average
            pct_unit_changes = [(u - avg_units) / avg_units * 100 for u in data["units"]]
            pct_price_changes = [(p - avg_price) / avg_price * 100 for p in data["prices"]]
            
            # Average % changes
            avg_pct_unit_change = statistics.mean([abs(x) for x in pct_unit_changes]) if pct_unit_changes else 0
            avg_pct_price_change = statistics.mean([abs(x) for x in pct_price_changes]) if pct_price_changes else 0
            
            # Price Elasticity
            if avg_pct_price_change > 0:
                elasticity = avg_pct_unit_change / avg_pct_price_change
            else:
                elasticity = 0
            
            results.append({
                "product": product,
                "avg_units": round(avg_units, 2),
                "avg_price": round(avg_price, 2),
                "pct_unit_variation": round(avg_pct_unit_change, 2),
                "pct_price_variation": round(avg_pct_price_change, 2),
                "elasticity": round(elasticity, 3),
                "interpretation": self._interpret_elasticity(elasticity)
            })
        
        # Sort by elasticity (highest first)
        results.sort(key=lambda x: x["elasticity"], reverse=True)
        results = results[:limit]
        
        return {
            "success": True,
            "analysis_type": "price_elasticity",
            "year": year,
            "data": results,
            "sql_query": sql,
            "explanation": f"Price elasticity calculated for {len(results)} products in {year}. Higher elasticity = more price sensitive.",
            "x_axis": "product",
            "y_axis": "elasticity"
        }
    
    def _interpret_elasticity(self, e: float) -> str:
        """Interpret elasticity value"""
        if e > 1.5:
            return "Highly elastic (very price sensitive)"
        elif e > 1:
            return "Elastic (price sensitive)"
        elif e > 0.5:
            return "Unit elastic"
        else:
            return "Inelastic (not price sensitive)"
    
    def analyze_cannibalization(self, table_name: str, params: Dict) -> Dict[str, Any]:
        """
        Analyze product cannibalization
        Find products where one's growth correlates with another's decline
        """
        year = params.get("year", "2023")
        limit = params.get("limit", 20)
        
        cols = self._get_column_names(table_name)
        product_col = cols["product"]
        units_col = cols["units"]
        date_col = cols["date"]
        
        conn = get_conn()
        cursor = conn.cursor()
        
        # Get product performance by period
        sql = f"""
        SELECT [{product_col}] AS product,
               [{date_col}] AS period,
               SUM([{units_col}]) AS total_units
        FROM [{table_name}]
        WHERE [{date_col}] LIKE '%{year}%'
          AND [{product_col}] IS NOT NULL
        GROUP BY [{product_col}], [{date_col}]
        ORDER BY [{product_col}], [{date_col}]
        """
        
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {"success": False, "error": "No data found", "data": []}
        
        # Group by product
        products = {}
        all_periods = set()
        for product, period, units in rows:
            if product not in products:
                products[product] = {}
            products[product][period] = float(units) if units else 0
            all_periods.add(period)
        
        periods = sorted(all_periods)
        
        # Calculate growth rates
        growth_rates = {}
        for product, period_data in products.items():
            values = [period_data.get(p, 0) for p in periods]
            if len(values) >= 2 and values[0] > 0:
                growth = (values[-1] - values[0]) / values[0] * 100
                growth_rates[product] = {
                    "growth_pct": growth,
                    "start_units": values[0],
                    "end_units": values[-1]
                }
        
        # Find potential cannibalization pairs
        # Products that grew while similar products declined
        growing = [(p, d) for p, d in growth_rates.items() if d["growth_pct"] > 10]
        declining = [(p, d) for p, d in growth_rates.items() if d["growth_pct"] < -10]
        
        results = []
        for gp, gd in growing[:10]:
            for dp, dd in declining[:10]:
                if gp != dp:
                    results.append({
                        "growing_product": gp,
                        "growing_pct": round(gd["growth_pct"], 1),
                        "declining_product": dp,
                        "declining_pct": round(dd["growth_pct"], 1),
                        "potential_cannibalization": "High" if abs(gd["growth_pct"]) > 20 and abs(dd["growth_pct"]) > 20 else "Medium"
                    })
        
        results = results[:limit]
        
        return {
            "success": True,
            "analysis_type": "cannibalization",
            "year": year,
            "data": results,
            "sql_query": sql,
            "explanation": f"Found {len(results)} potential cannibalization pairs. Growing products may be taking sales from declining ones.",
            "summary": {
                "total_growing": len(growing),
                "total_declining": len(declining)
            }
        }
    
    def calculate_correlation(self, table_name: str, params: Dict) -> Dict[str, Any]:
        """Calculate correlation between two metrics"""
        # Simplified correlation calculation
        return {
            "success": False,
            "error": "Correlation analysis requires specifying two metrics",
            "data": []
        }
    
    def calculate_yoy_growth(self, table_name: str, params: Dict) -> Dict[str, Any]:
        """Calculate year-over-year growth for products"""
        cols = self._get_column_names(table_name)
        product_col = cols["product"]
        units_col = cols["units"]
        date_col = cols["date"]
        limit = params.get("limit", 50)
        
        conn = get_conn()
        cursor = conn.cursor()
        
        sql = f"""
        WITH yearly AS (
            SELECT [{product_col}] AS product,
                   CASE 
                       WHEN [{date_col}] LIKE '%2024%' THEN '2024'
                       WHEN [{date_col}] LIKE '%2023%' THEN '2023'
                       WHEN [{date_col}] LIKE '%2022%' THEN '2022'
                   END AS year,
                   SUM([{units_col}]) AS total
            FROM [{table_name}]
            WHERE [{product_col}] IS NOT NULL
            GROUP BY [{product_col}], year
        )
        SELECT 
            cy.product,
            cy.total AS current_year,
            py.total AS prior_year,
            ROUND((cy.total - COALESCE(py.total, 0)) * 100.0 / NULLIF(py.total, 0), 1) AS yoy_growth
        FROM yearly cy
        LEFT JOIN yearly py ON cy.product = py.product 
            AND CAST(cy.year AS INT) = CAST(py.year AS INT) + 1
        WHERE cy.year = '2024' OR cy.year = '2023'
        ORDER BY yoy_growth DESC
        LIMIT {limit}
        """
        
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        
        results = [
            {"product": r[0], "current_year": r[1], "prior_year": r[2], "yoy_growth": r[3]}
            for r in rows if r[3] is not None
        ]
        
        return {
            "success": True,
            "analysis_type": "yoy_growth",
            "data": results,
            "sql_query": sql,
            "explanation": f"Year-over-year growth for {len(results)} products"
        }


# Singleton
_analytics_agent = None

def get_analytics_agent() -> AnalyticsAgent:
    global _analytics_agent
    if _analytics_agent is None:
        _analytics_agent = AnalyticsAgent()
    return _analytics_agent