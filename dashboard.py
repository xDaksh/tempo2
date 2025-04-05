import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
import threading
from chatbot import chat_with_bot
from nudges import get_gamified_nudges, get_category_warnings
from razorpay_realtime import fetch_latest_payments

# --- PAGE CONFIG --- #
st.set_page_config(page_title="AI Finance Assistant", layout="wide")

# --- SPLASH SCREEN --- #
if "splash_shown" not in st.session_state:
    splash = st.empty()
    with splash.container():
        st.image("splash.png", use_container_width=True)
        time.sleep(3)
    splash.empty()
    st.session_state["splash_shown"] = True

# --- DATA LOADING --- #
@st.cache_data
def load_data():
    df_local = pd.read_csv("mock_transactions_detailed.csv", parse_dates=["datetime"])
    try:
        df_rzp = fetch_latest_payments()
        df_combined = pd.concat([df_local, df_rzp], ignore_index=True)
        return df_combined
    except:
        return df_local

# --- FILE UPLOAD --- #
uploaded_file = st.file_uploader("Upload your transaction data (CSV)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["datetime"])
    st.success("✅ File uploaded and loaded successfully!")
else:
    df = load_data()

# --- SIDEBAR FILTERS --- #
with st.sidebar:
    st.title("⚙ Dashboard Settings")

    # Date Filter
    date_range = st.date_input(
        "📅 Date Range",
        [df["datetime"].min().date(), df["datetime"].max().date()]
    )

    # Type Filter
    selected_type = st.multiselect(
        "📂 Type",
        options=df["type"].unique(),
        default=df["type"].unique()
    )

    # Category Filter
    selected_category = st.multiselect(
        "🏷 Category",
        options=df["category"].unique(),
        default=df["category"].unique()
    )

# --- BUDGET SETTINGS --- #
if "budget" not in st.session_state:
    st.session_state.budget = 10000

category_budgets = {}
for cat in df['category'].unique():
    if f"budget_{cat}" not in st.session_state:
        st.session_state[f"budget_{cat}"] = 1000
    category_budgets[cat] = st.session_state[f"budget_{cat}"]

# --- FILTERED DATA --- #
filtered_df = df[
    (df["type"].isin(selected_type)) &
    (df["category"].isin(selected_category)) &
    (df["datetime"].dt.date >= date_range[0]) &
    (df["datetime"].dt.date <= date_range[1])
]

# --- NAVIGATION --- #
st.sidebar.title("🔍 Navigation")
pages = {
    "🏠 Dashboard": "dashboard",
    "📊 Expense Forecasting": "forecasting",
    "📅 Monthly Spending": "monthly",
    "📆 Weekly Spending": "weekly",
    "📂 Category Analysis": "category",
    "💬 AI Chatbot": "chatbot"
}
selected_page = st.sidebar.radio("Select a section:", list(pages.keys()))

# --- PAGE CONTENT --- #
if pages[selected_page] == "dashboard":
    st.title("💰 AI Finance Assistant Dashboard")

    # Quick Summary
    st.subheader("📈 Quick Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spent", f"₹{filtered_df['amount'].sum():,.2f}")
    col2.metric("Transactions", f"{len(filtered_df)}")
    col3.metric("Avg. per Transaction", f"₹{filtered_df['amount'].mean():,.2f}")

    # Monthly Budget Progress
    current_month = pd.Timestamp.now().strftime('%Y-%m')
    df_this_month = filtered_df[filtered_df["datetime"].dt.to_period('M').astype(str) == current_month]
    spent_this_month = df_this_month["amount"].sum()
    progress = min(spent_this_month / st.session_state.budget, 1.0)

    st.subheader("📊 Monthly Budget Progress")
    st.progress(progress)
    col1, col2 = st.columns(2)
    col1.metric("Spent This Month", f"₹{spent_this_month:,.0f}")
    col2.metric("Remaining Budget", f"₹{st.session_state.budget - spent_this_month:,.0f}")

elif pages[selected_page] == "forecasting":
    st.title("📉 Expense Forecasting")
    monthly_expenses = filtered_df.groupby(filtered_df["datetime"].dt.to_period("M"))['amount'].sum().reset_index()
    if len(monthly_expenses) >= 2:
        monthly_expenses['month_num'] = range(1, len(monthly_expenses) + 1)
        X = monthly_expenses[['month_num']]
        y = monthly_expenses['amount']
        model = LinearRegression()
        model.fit(X, y)
        next_month = np.array([[monthly_expenses['month_num'].max() + 1]])
        prediction = model.predict(next_month)[0]
        st.info(f"📅 Predicted expense for next month: ₹{prediction:,.0f}")

elif pages[selected_page] == "chatbot":
    st.title("💬 AI Chatbot")
    user_input = st.chat_input("Ask me about your finances")
    if user_input:
        response = chat_with_bot(user_input, filtered_df)
        st.success(response)

# --- AUTO REFRESH --- #
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
elif time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# --- REAL-TIME RAZORPAY PAYMENT TRACKING FUNCTION --- #
def start_realtime_tracking():
    while True:
        print("Tracking real-time payments...")  # Replace with actual tracking logic
        time.sleep(10)  # Simulate a delay for periodic update

# --- AUTHENTICATION --- #
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("🔐 Login to Continue")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "shatwik" and hash_password(password) == hash_password("12903478"):
                st.session_state.logged_in = True
                st.success("Login successful!")
            else:
                st.error("Invalid credentials")
        st.stop()

check_login()

# --- START BACKGROUND RAZORPAY TRACKER --- #
if "razorpay_tracker_started" not in st.session_state:
    st.session_state.razorpay_tracker_started = True
    threading.Thread(target=start_realtime_tracking, daemon=True).start()


# --- REAL-TIME ALERT FOR NEW TRANSACTIONS --- #
@st.cache_data
def get_latest_transaction_time(df):
    return df['datetime'].max()

latest_time = get_latest_transaction_time(df)
if "last_seen_txn_time" not in st.session_state:
    st.session_state.last_seen_txn_time = latest_time
elif latest_time > st.session_state.last_seen_txn_time:
    st.balloons()
    st.success("🎉 New transaction detected!")
    st.session_state.last_seen_txn_time = latest_time

# --- OPTIONAL TAGGING UI --- #
st.subheader("🏷 Tag Unknown Categories")

# Define category-to-need/want mapping
need_want_map = {
    "Shopping": "Want",
    "Transport": "Need",
    "Grocery": "Need",
    "Health": "Need",
    "Entertainment": "Want",
    "Education": "Need",
    "food": "Want",
    "Gaming": "Want"
}

untagged = df[df["category"] == ""].copy()

if not untagged.empty:
    for i, row in untagged.iterrows():
        col1, col2, col3 = st.columns([2, 2, 4])
        with col1:
            date_range = st.sidebar.date_input("Select Date Range", value=[min_date, max_date])


import hashlib
import os
from auth import auth_flow