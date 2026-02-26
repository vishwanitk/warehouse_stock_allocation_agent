#!/usr/bin/env python
# coding: utf-8

import os
from dotenv import load_dotenv
load_dotenv()





import pandas as pd
import numpy as np


def generate_raw_data():
    """
    Generate synthetic sales data for multiple stores.
    
   
    
    Returns:
    - dict with 'sales_df' 
    """
    np.random.seed(42)

    stores = [f"Store_{i}" for i in range(1, 9)]
    days = 120
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)

    data = []

    for store in stores:
        base_demand = np.random.randint(20, 60)
        trend = np.linspace(0, 10, days)
        seasonality = 10 * np.sin(np.linspace(0, 3*np.pi, days))
        noise = np.random.normal(0, 5, days)

        sales = base_demand + trend + seasonality + noise
        sales = np.maximum(0, sales)

        for d, s in zip(dates, sales):
            data.append({
                "date": d,
                "store": store,
                "sales": round(s)
            })

    df = pd.DataFrame(data)
    
    constraints_df = pd.DataFrame({
    "store": [f"Store_{i}" for i in range(1, 9)],
    "min_stock": [50, 50, 50, 50, 50, 50, 60, 50],
    "max_stock": [500, 500, 500, 500, 500, 500, 500, 500],
    "priority":  [5, 7, 4, 3, 2, 1, 8, 6]
      })
    
    raw_data = {
        "sales_df": df,
        "warehouse_stock": 1000,
        "constraints_df": constraints_df 
        
    }

    return raw_data


generate_raw_data()


from typing import TypedDict, Dict

class State(TypedDict, total=False):
    raw_data: Dict
    cleaned_data: pd.DataFrame
    constraints_df: pd.DataFrame
    warehouse_stock: int
    total_target_required: float
    total_min_required: float
    remaining_stock: float
    forecast: Dict
    safety_stock: Dict
    target_inventory: Dict
    allocation_plan: Dict
    final_output: str
    forecast_adjustment_factor: float


def clean_data_node(state: State):
    df = state["raw_data"]["sales_df"]

    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"])
  

    df = df.sort_values(["store", "date"])

    

    state["cleaned_data"] = df
    return state





from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import json

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=500)

def llm_summary_node(state: State):
    allocation = state["allocation_plan"]
    warehouse_stock = state["raw_data"]["warehouse_stock"]
    total_required = round(state["total_target_required"])

    # Convert allocation dict to JSON string (integers only)
    allocation_json = json.dumps(allocation, indent=2)
    forecast_json= json.dumps(state["forecast"], indent=2)
    prompt = f"""
You are a retail supply chain strategist.

Do NOT recalculate anything. Use the allocation provided below exactly as-is.

Allocation dictionary (all values are integers):
Forecast: {forecast_json}

give some 2 lines gap 
Allocation plan: {allocation_json}

Total required stock across stores: {int(total_required)}
Warehouse stock available: {int(warehouse_stock)}

STRICT OUTPUT FORMAT:

Return ONLY a JSON object with these keys:
1. "warehouse_stock: f"{warehouse_stock}"
2.  "total_required": f"{total_required}"
3. "forecast" : dictionary of store -> rounder forecasted units (integers)
3. "allocation" : dictionary of store -> allocated units (integers)
4. "warehouse_remaining" : integer, remaining stock in warehouse


Do NOT add explanations, reasoning, or extra commentary.
Do NOT modify the numbers. Ensure JSON is valid.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # ⚡ Keep final output only
    final_state = {
        "final_output": response.content,
        "allocation_plan": allocation
    }
    return final_state


from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_node(state: State):
    df = state["cleaned_data"]  # should contain 'store', 'date', 'sales'

    forecasts = {}
    forecast_adjustment_factor = 1
    for store in df["store"].unique():
        store_df = df[df["store"] == store].copy()

        # 1️⃣ Sort by date
        store_df = store_df.sort_values("date")

        # 2️⃣ Set datetime index
        store_df = store_df.set_index("date")
        store_df.index = pd.DatetimeIndex(store_df.index)
        store_df = store_df.asfreq("D")  # daily frequency

        # 3️⃣ Fit model
        model = ExponentialSmoothing(
            store_df["sales"],
            trend="add",
            seasonal=None
        ).fit()

        # 4️⃣ Forecast 7-day demand
        forecast = model.forecast(7).sum()  # total demand for next 7 days
        forecasts[store] = forecast * forecast_adjustment_factor

    state["forecast"] = forecasts
    state["forecast_adjustment_factor"]= forecast_adjustment_factor
    return state


import numpy as np
from scipy.stats import norm

def safety_stock_node(state: State):
    df = state["cleaned_data"]
    forecasts = state["forecast"]

    safety = {}
    service_level = 0.95
    Z = norm.ppf(service_level)

    for store in forecasts:
        store_sales = df[df["store"] == store]["sales"]
        sigma = np.std(store_sales)
        lead_time = 7

        ss = Z * sigma * np.sqrt(lead_time)
        safety[store] = ss

    state["safety_stock"] = safety
    return state


def target_inventory_node(state: State):
    forecasts = state["forecast"]
    safety = state["safety_stock"]
    constraints_df = state["raw_data"]["constraints_df"]

    # Convert dataframe to dictionary keyed by store
    constraints = constraints_df.set_index("store").to_dict("index")

    target = {}

    for store in forecasts:
        target_value = forecasts[store] + safety[store]

        target[store] = {
            "target": target_value,
            "min_stock": constraints[store]["min_stock"],
            "max_stock": constraints[store]["max_stock"],
            "priority": constraints[store]["priority"]
        }

    state["target_inventory"] = target
    return state



def allocation_node(state: State):

    stores = state["target_inventory"]
    warehouse_stock = state["raw_data"]["warehouse_stock"]

    allocation = {}

    # ------------------------------------------------
    # 1️⃣ Calculate totals
    # ------------------------------------------------
    total_target_required = sum(store["target"] for store in stores.values())
    total_min_required = sum(store["min_stock"] for store in stores.values())

    # ------------------------------------------------
    # 2️⃣ Check feasibility
    # ------------------------------------------------
    if warehouse_stock < total_min_required:
        raise ValueError(
            f"Warehouse stock ({warehouse_stock}) "
            f"is below total minimum required ({total_min_required})"
        )

    # ------------------------------------------------
    # 3️⃣ Allocate minimum stock first
    # ------------------------------------------------
    for store_name, store_data in stores.items():
        allocation[store_name] = store_data["min_stock"]

    remaining_stock = warehouse_stock - total_min_required

    # ------------------------------------------------
    # 4️⃣ Allocate remaining by priority (high → low)
    # ------------------------------------------------
    sorted_stores = sorted(
        stores.items(),
        key=lambda x: x[1]["priority"],
        reverse=True
    )

    for store_name, store_data in sorted_stores:
        if remaining_stock <= 0:
            break

        current_alloc = allocation[store_name]
        max_allowed = store_data["max_stock"]
        target_value = store_data["target"]

        # Can't exceed either target or max_stock
        upper_limit = min(max_allowed, target_value)

        possible_extra = upper_limit - current_alloc

        if possible_extra > 0:
            extra = min(possible_extra, remaining_stock)
            allocation[store_name] += int(extra)
            remaining_stock -= int(extra)

    # ------------------------------------------------
    # 5️⃣ Save results
    # ------------------------------------------------
    state["allocation_plan"] = allocation
    state["total_target_required"] = total_target_required
    state["total_min_required"] = total_min_required
    state["remaining_stock"] = remaining_stock

    return state


from langgraph.graph import StateGraph

builder = StateGraph(State)

builder.add_node("clean", clean_data_node)
builder.add_node("forecast", forecast_node)
builder.add_node("safety", safety_stock_node)
builder.add_node("target", target_inventory_node)
builder.add_node("allocate", allocation_node)
builder.add_node("llm_summary", llm_summary_node)

builder.set_entry_point("clean")

builder.add_edge("clean", "forecast")
builder.add_edge("forecast", "safety")
builder.add_edge("safety", "target")
builder.add_edge("target", "allocate")
builder.add_edge("allocate", "llm_summary")

graph = builder.compile()


raw_data = generate_raw_data()

initial_state = {
    "raw_data": raw_data,
}

result = graph.invoke(initial_state)


print(result["final_output"])





df_raw=generate_raw_data()


import seaborn as sns


# Suppose your dict is called `raw_data`
df = raw_data['sales_df']

print(df.head())



sns.lineplot(data=df, x="date", y="sales", hue="store")


raw_data['warehouse_stock']


raw_data




