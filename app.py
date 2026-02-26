import streamlit as st
import json
from project import graph, generate_raw_data

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="AI Supply Chain Allocation",
    layout="wide"
)

st.title("📦 AI-Powered Warehouse Allocation System")



st.markdown("Adjust parameters and run the LangGraph pipeline.")

# ----------------------------------
# User Inputs
# ----------------------------------
warehouse_stock = st.number_input(
    "Warehouse Stock",
    min_value=0,
    value=1000,
    step=50
)

forecast_adjustment_factor = st.slider(
    "Forecast Adjustment Factor",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1
)

# ----------------------------------
# Run Graph Button
# ----------------------------------
if st.button("🚀 Run Allocation Graph"):

    with st.spinner("Running Supply Chain Optimization..."):

        # Generate raw data
        raw_data = generate_raw_data()

        st.subheader("Sample Sales Data")
        st.dataframe(raw_data["sales_df"].head())

        st.subheader("Constraints Data")
        st.dataframe(raw_data["constraints_df"])


        # Override warehouse stock from user input
        raw_data["warehouse_stock"] = warehouse_stock

        # Initial state
        initial_state = {
            "raw_data": raw_data,
            "forecast_adjustment_factor": forecast_adjustment_factor
        }

        try:
            result = graph.invoke(initial_state)

            st.success("Graph Execution Completed")

            # ----------------------------------
            # Display Allocation Table
            # ----------------------------------
            st.subheader("📊 Allocation Plan")

            allocation = result["allocation_plan"]
            allocation_df = (
                st.session_state.get("allocation_df")
                if False else None
            )

            import pandas as pd
            allocation_df = pd.DataFrame(
                list(allocation.items()),
                columns=["Store", "Allocated Units"]
            )

            st.dataframe(allocation_df, use_container_width=True)

            # ----------------------------------
            # Display LLM Output
            # ----------------------------------
            st.subheader("🧠 AI Summary")

            try:
                output_json = json.loads(result["final_output"])
                st.json(output_json)
            except:
                st.text(result["final_output"])

        except Exception as e:
            st.error(f"Error: {e}")
