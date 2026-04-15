import streamlit as st
import pandas as pd
import time

# 1. Page Configuration
st.set_page_config(page_title="Mask Compliance Dashboard", layout="wide")
st.title("😷 Real-Time Face Mask Analytics")
st.markdown("This dashboard tracks live data from the OpenCV Computer Vision model.")

# 2. Function to load the data from your CSV
def load_data():
    try:
        # Read the CSV file you created in the previous step
        df = pd.read_csv("mask_analytics_log.csv")
        return df
    except FileNotFoundError:
        st.error("Waiting for data... Please run your detect_mask_webcam.py script first!")
        return pd.DataFrame(columns=['Timestamp', 'Mask_Count', 'No_Mask_Count', 'Total_People'])

# 3. Create a placeholder that we will update in a loop
placeholder = st.empty()

# 4. Loop to keep the dashboard updating automatically
while True:
    df = load_data()
    
    if not df.empty:
        with placeholder.container():
            # Calculate total metrics for the top of the dashboard
            latest_masks = df['Mask_Count'].iloc[-1]
            latest_no_masks = df['No_Mask_Count'].iloc[-1]
            total_scanned = df['Total_People'].sum()

            # Display large metric numbers
            col1, col2, col3 = st.columns(3)
            col1.metric("Currently Masked", latest_masks)
            col2.metric("Currently Unmasked", latest_no_masks)
            col3.metric("Total People Logged", total_scanned)

            st.markdown("---")
            st.subheader("Compliance Timeline")
            
            # Prepare the data for the chart (set Timestamp as the index)
            chart_data = df.set_index('Timestamp')[['Mask_Count', 'No_Mask_Count']]
            
            # Draw a beautiful line chart
            st.line_chart(chart_data, color=["#00FF00", "#FF0000"]) # Green for masks, Red for no masks

    # Wait 3 seconds before refreshing the data
    time.sleep(3)