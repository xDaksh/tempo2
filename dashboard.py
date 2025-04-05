import streamlit as st
st.set_page_config(page_title="AI Finance Assistant", layout="wide")
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from chatbot import chat_with_bot
from nudges import get_gamified_nudges, get_category_warnings
import hashlib
import os
from auth import auth_flow
import numpy as np
from sklearn.linear_model import LinearRegression
import threading
from razorpay_realtime import fetch_latest_payments

# --- REAL-TIME RAZORPAY PAYMENT TRACKING FUNCTION --- #
def start_realtime_tracking():
    # This is a placeholder function.
    # You can replace it with your actual logic to track real-time Razorpay payments.
    while True:
        print("Tracking real-time payments...")  # Replace with actual tracking logic
        time.sleep(10)  # Simulate a delay for periodic update
        
@st.cache_data(ttl=60)
def load_data():
    df_local = pd.read_csv("mock_transactions_detailed.csv", parse_dates=["datetime"])
    df_rzp = fetch_latest_payments()
    df_combined = pd.concat([df_local, df_rzp], ignore_index=True)
    return df_combined


# --- AUTHENTICATION --- #
auth_flow()

# --- START BACKGROUND RAZORPAY TRACKER --- #
if "razorpay_tracker_started" not in st.session_state:
    st.session_state.razorpay_tracker_started = True
    threading.Thread(target=start_realtime_tracking, daemon=True).start()

# --- AUTO REFRESH EVERY 60 SECONDS --- #
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
else:
    if time.time() - st.session_state.last_refresh > 60:
        st.session_state.last_refresh = time.time()
        st.experimental_rerun()

# --- LOGIN SYSTEM --- #
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

# --- LOADING SCREEN --- #
with st.spinner("🧙‍♂ Summoning your Gringotts vault... Please wait..."):
    time.sleep(2)

# --- LOAD DATA --- #
@st.cache_data
def load_data():
    df_local = pd.read_csv("mock_transactions_detailed.csv", parse_dates=["datetime"])
    try:
        df_rzp = pd.read_csv("razorpay_payments.csv", parse_dates=["datetime"])
        df_rzp["type"] = "expense"
        df_rzp["category"] = ""
        df_combined = pd.concat([df_local, df_rzp], ignore_index=True)
        return df_combined
    except FileNotFoundError:
        return df_local

st.sidebar.subheader("📁 Upload Your Transactions (CSV)")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["datetime"])
    st.success("✅ File uploaded and loaded successfully!")
else:
    df = load_data()

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
untagged = df[df["category"] == ""].copy()
if not untagged.empty:
    for i, row in untagged.iterrows():
        col1, col2, col3 = st.columns([2, 2, 4])
        with col1:
            st.write(f"🧾 {row['app']} - ₹{row['amount']}")
        with col2:
            category = st.selectbox(f"Category for {row['app']}", options=[
                "Shopping", "Transport", "Grocery", "Bills", "Entertainment", "Other"
            ], key=f"tag_{i}")
        with col3:
            if st.button("Save", key=f"save_{i}"):
                df.at[i, "category"] = category
                df.to_csv("mock_transactions_detailed.csv", index=False)
                st.success("✅ Category tagged!")

# --- TITLE --- #
st.title("💰 AI Finance Assistant Dashboard")

# --- SIDEBAR SETTINGS --- #
with st.sidebar:
    st.title("⚙️ Dashboard Settings")

    # 🔐 Razorpay API Login
    with st.expander("🔑 Razorpay API Login", expanded=False):
        if "razorpay_key" not in st.session_state:
            st.session_state["razorpay_key"] = ""
            st.session_state["razorpay_secret"] = ""

        api_key = st.text_input("🗝 API Key", type="password", value=st.session_state["razorpay_key"])
        api_secret = st.text_input("🔏 API Secret", type="password", value=st.session_state["razorpay_secret"])

        if st.button("🔓 Authenticate", use_container_width=True):
            if api_key and api_secret:
                st.session_state["razorpay_key"] = api_key
                st.session_state["razorpay_secret"] = api_secret
                st.success("✅ Authentication Successful!")
            else:
                st.error("⚠ Please enter both API Key and Secret!")

    # 📊 Filters Section
    with st.expander("🔍 Filters", expanded=False):
        selected_type = st.multiselect("📂 Type of Expense", df['type'].unique(), default=df['type'].unique())
        selected_category = st.multiselect("🏷 Category", df['category'].unique(), default=df['category'].unique())
        date_range = st.date_input("📅 Date Range", [df["datetime"].min().date(), df["datetime"].max().date()])

    # 🎯 Budget Settings
    with st.expander("💰 Budget Settings", expanded=False):
        if "budget" not in st.session_state:
            st.session_state.budget = 10000

        new_budget = st.number_input("📌 Set your budget (₹)", min_value=0, value=st.session_state.budget, step=500)
        if new_budget != st.session_state.budget:
            st.session_state.budget = new_budget
            st.success(f"✅ Budget updated to ₹{new_budget}")

    # 📊 Category-wise Budget
    with st.expander("📊 Category Budgets", expanded=False):
        category_budgets = {}
        for cat in df['category'].unique():
            category_budgets[cat] = st.number_input(f"📂 {cat} Budget (₹)", min_value=0, value=1000, step=100)

    # 🔴 Real-time Payments Monitoring
    with st.expander("💳 Real-time Payments", expanded=False):
        if "razorpay_key" in st.session_state and st.session_state["razorpay_key"]:
            if st.button("🔄 Fetch Latest Transactions", use_container_width=True):
                st.success("📡 Fetching real-time payments...")
                df_live = fetch_latest_payments()  # Call function from razorpay_realtime.py
                df = pd.concat([df, df_live], ignore_index=True)
                st.success("✅ Updated with new transactions!")
        else:
            st.warning("⚠ Please authenticate Razorpay to enable real-time tracking!")

    # --- SIDEBAR NAVIGATION --- #
    st.divider()
    st.subheader("🔍 Navigation")
    menu_option = st.radio(
        "Select a section:",
        [
            "🏠 Dashboard",
            "📊 Expense Forecasting",
            "🔍 Category-wise Expense Forecasting",
            "🏆 Achievement Nudges",
            "⚠ Budget Warnings",
            "💬 AI Chatbot"
        ]
    )

# --- MAIN PAGE CONTENT BASED ON SELECTION --- #
if menu_option == "🏠 Dashboard":
    st.title("💰 AI Finance Assistant Dashboard")
    
    # Ensure filtered_df is defined
    if "filtered_df" not in locals():
        filtered_df = df  # Assign it to your main DataFrame if not already filtered
    
    # Quick Summary
    st.subheader("📈 Quick Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spent", f"₹{filtered_df['amount'].sum():,.2f}")
    col2.metric("Transactions", f"{len(filtered_df)}")
    col3.metric("Avg. per Transaction", f"₹{filtered_df['amount'].mean():,.2f}")

    # Monthly Budget Progress
    st.subheader("📊 Monthly Budget Progress")
    st.progress(progress)
    col1, col2 = st.columns(2)
    col1.metric("Spent This Month", f"₹{spent_this_month:,.0f}")
    col2.metric("Remaining Budget", f"₹{budget - spent_this_month:,.0f}")

elif menu_option == "📊 Expense Forecasting":
    st.subheader("📉 Expense Forecasting")

    # Ensure monthly_expenses is defined
    if "monthly_expenses" not in locals():
        monthly_expenses = []  # Initialize properly

    if len(monthly_expenses) >= 2:
        last_month = monthly_expenses.iloc[-2]["amount"]
        current_month = monthly_expenses.iloc[-1]["amount"]
        growth_rate = (current_month - last_month) / last_month if last_month > 0 else 0
        prediction = current_month * (1 + growth_rate)
        
        st.info(f"📅 Predicted expense for next month: ₹{prediction:,.0f}")
        
        if prediction > budget:
            st.warning("🚨 Your next month's expenses may exceed your set budget!")
    else:
        st.info("📉 Not enough data to generate forecast.")

elif menu_option == "🔍 Category-wise Expense Forecasting":
    st.subheader("🔍 Category-wise Expense Forecasting")

    # Ensure future_forecasts is defined
    if "future_forecasts" not in locals():
        future_forecasts = {}  # Initialize to avoid errors
    
    for cat, forecast in future_forecasts.items():
        cat_budget = category_budgets.get(cat, 0)
        forecast_msg = f"📌 *{cat}*: Forecasted ₹{forecast:.0f} / Budget ₹{cat_budget}"
        if forecast > cat_budget:
            st.warning(f"🚨 {forecast_msg} — Likely to overspend!")
        else:
            st.info(f"✅ {forecast_msg} — Looks safe.")

elif menu_option == "🏆 Achievement Nudges":
    st.subheader("🏆 Achievement Nudges")
    badges = get_gamified_nudges(df_this_month, budget)
    for badge in badges:
        st.success(badge)

elif menu_option == "⚠ Budget Warnings":
    st.subheader("⚠ Category Budget Warnings")
    category_warnings = get_category_warnings(df_this_month, category_budgets)
    for warning in category_warnings:
        st.warning(warning)

elif menu_option == "💬 AI Chatbot":
    st.subheader("💬 Ask Your Assistant")
    user_input = st.chat_input("Talk to your finance assistant")
    if user_input:
        response = chat_with_bot(user_input, filtered_df)
        st.success(response)
if menu_option == "🏠 Dashboard":
    st.title("💰 AI Finance Assistant Dashboard")
    
    # Ensure filtered_df is defined
    if "filtered_df" not in locals():
        filtered_df = df  # Assign it to your main DataFrame if not already filtered
    
    # Quick Summary
    st.subheader("📈 Quick Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spent", f"₹{filtered_df['amount'].sum():,.2f}")
    col2.metric("Transactions", f"{len(filtered_df)}")
    col3.metric("Avg. per Transaction", f"₹{filtered_df['amount'].mean():,.2f}")

    # Monthly Budget Progress
    st.subheader("📊 Monthly Budget Progress")
    st.progress(progress)
    col1, col2 = st.columns(2)
    col1.metric("Spent This Month", f"₹{spent_this_month:,.0f}")
    col2.metric("Remaining Budget", f"₹{budget - spent_this_month:,.0f}")

elif menu_option == "📊 Expense Forecasting":
    st.subheader("📉 Expense Forecasting")

    # Ensure monthly_expenses is defined
    if "monthly_expenses" not in locals():
        monthly_expenses = []  # Initialize properly

    if len(monthly_expenses) >= 2:
        last_month = monthly_expenses.iloc[-2]["amount"]
        current_month = monthly_expenses.iloc[-1]["amount"]
        growth_rate = (current_month - last_month) / last_month if last_month > 0 else 0
        prediction = current_month * (1 + growth_rate)
        
        st.info(f"📅 Predicted expense for next month: ₹{prediction:,.0f}")
        
        if prediction > budget:
            st.warning("🚨 Your next month's expenses may exceed your set budget!")
    else:
        st.info("📉 Not enough data to generate forecast.")

elif menu_option == "🔍 Category-wise Expense Forecasting":
    st.subheader("🔍 Category-wise Expense Forecasting")

    # Ensure future_forecasts is defined
    if "future_forecasts" not in locals():
        future_forecasts = {}  # Initialize to avoid errors
    
    for cat, forecast in future_forecasts.items():
        cat_budget = category_budgets.get(cat, 0)
        forecast_msg = f"📌 *{cat}*: Forecasted ₹{forecast:.0f} / Budget ₹{cat_budget}"
        if forecast > cat_budget:
            st.warning(f"🚨 {forecast_msg} — Likely to overspend!")
        else:
            st.info(f"✅ {forecast_msg} — Looks safe.")

elif menu_option == "🏆 Achievement Nudges":
    st.subheader("🏆 Achievement Nudges")
    badges = get_gamified_nudges(df_this_month, budget)
    for badge in badges:
        st.success(badge)

elif menu_option == "⚠ Budget Warnings":
    st.subheader("⚠ Category Budget Warnings")
    category_warnings = get_category_warnings(df_this_month, category_budgets)
    for warning in category_warnings:
        st.warning(warning)

elif menu_option == "💬 AI Chatbot":
    st.subheader("💬 Ask Your Assistant")
    user_input = st.chat_input("Talk to your finance assistant")
    if user_input:
        response = chat_with_bot(user_input, filtered_df)
        st.success(response)




# --- FILTERED DATA --- #
filtered_df = df[
    (df["type"].isin(selected_type)) &
    (df["category"].isin(selected_category)) &
    (df["datetime"].dt.date >= date_range[0]) &
    (df["datetime"].dt.date <= date_range[1])
]

# --- SUMMARY METRICS --- #
st.subheader("📈 Quick Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Spent", f"₹{filtered_df['amount'].sum():,.2f}")
col2.metric("Transactions", f"{len(filtered_df)}")
col3.metric("Avg. per Transaction", f"₹{filtered_df['amount'].mean():,.2f}")

# --- BUDGET PROGRESS --- #
current_month = pd.Timestamp.now().strftime('%Y-%m')
df_this_month = filtered_df[filtered_df["datetime"].dt.to_period('M').astype(str) == current_month]
spent_this_month = df_this_month["amount"].sum()
budget = st.session_state.budget
progress = min(spent_this_month / budget, 1.0)

st.subheader("📊 Monthly Budget Progress")
st.progress(progress)
col1, col2 = st.columns(2)
col1.metric("Spent This Month", f"₹{spent_this_month:,.0f}")
col2.metric("Remaining Budget", f"₹{budget - spent_this_month:,.0f}")

# --- CATEGORY-WISE SPENT & REMAINING --- #
st.subheader("🧾 Category-wise Budget Tracking")
for cat in df['category'].unique():
    cat_spent = df_this_month[df_this_month['category'] == cat]['amount'].sum()
    cat_budget = category_budgets.get(cat, 0)
    cat_remaining = cat_budget - cat_spent
    st.write(f"{cat}: Spent ₹{cat_spent:.0f} / ₹{cat_budget} | Remaining: ₹{cat_remaining:.0f}")

# --- EXPENSE FORECASTING --- #
st.subheader("📉 Expense Forecasting")
monthly_expenses = df.groupby(df["datetime"].dt.to_period("M"))['amount'].sum().reset_index()
monthly_expenses['datetime'] = monthly_expenses['datetime'].dt.to_timestamp()
monthly_expenses['month_num'] = range(1, len(monthly_expenses) + 1)

if len(monthly_expenses) >= 2:
    X = monthly_expenses[['month_num']]
    y = monthly_expenses['amount']
    model = LinearRegression()
    model.fit(X, y)
    next_month = np.array([[monthly_expenses['month_num'].max() + 1]])
    prediction = model.predict(next_month)[0]

    st.info(f"📅 Predicted expense for next month: ₹{prediction:,.0f}")

    if prediction > budget:
        st.warning("🚨 Your next month's expenses may exceed your set budget!")
else:
    st.info("📉 Not enough data to generate forecast.")

# --- CATEGORY-WISE FORECASTING --- #
st.subheader("🔍 Category-wise Expense Forecasting")
future_forecasts = {}
for cat in df['category'].unique():
    cat_df = df[df['category'] == cat]
    monthly_cat = cat_df.groupby(cat_df["datetime"].dt.to_period("M"))['amount'].sum().reset_index()
    if len(monthly_cat) < 2:
        continue
    monthly_cat['datetime'] = monthly_cat['datetime'].dt.to_timestamp()
    monthly_cat['month_num'] = range(1, len(monthly_cat) + 1)
    X_cat = monthly_cat[['month_num']]
    y_cat = monthly_cat['amount']
    model = LinearRegression()
    model.fit(X_cat, y_cat)
    next_month_cat = np.array([[monthly_cat['month_num'].max() + 1]])
    forecast_cat = model.predict(next_month_cat)[0]
    future_forecasts[cat] = forecast_cat

for cat, forecast in future_forecasts.items():
    cat_budget = category_budgets.get(cat, 0)
    forecast_msg = f"📌 *{cat}*: Forecasted ₹{forecast:.0f} / Budget ₹{cat_budget}"
    if forecast > cat_budget:
        st.warning(f"🚨 {forecast_msg} — Likely to overspend!")
    else:
        st.info(f"✅ {forecast_msg} — Looks safe.")

# --- GAMIFIED BADGES --- #
def get_savings_badge(savings):
    if 1 <= savings <= 100:
        return "🥉 Bronze Saver - Good start!"
    elif 101 <= savings <= 500:
        return "🥈 Silver Saver - Nice job!"
    elif 501 <= savings <= 1000:
        return "🥇 Gold Saver - Great work!"
    elif 1001 <= savings <= 2000:
        return "🏅 Platinum Saver - You're killing it!"
    elif savings > 2000:
        return "💎 Diamond Saver - Legendary savings!"
    return None

savings = budget - spent_this_month
badge = get_savings_badge(savings)
if badge:
    st.success(f"🏆 {badge}")

# --- GAMIFIED NUDGES --- #
st.subheader("🏆 Achievement Nudges")
badges = get_gamified_nudges(df_this_month, budget)
for badge in badges:
    st.success(badge)

# --- CATEGORY BUDGET WARNINGS --- #
st.subheader("⚠ Category Budget Warnings")
category_warnings = get_category_warnings(df_this_month, category_budgets)
for warning in category_warnings:
    st.warning(warning)

# --- MONTHLY SPENDING --- #
st.subheader("📅 Monthly Spending")
filtered_df['month'] = filtered_df['datetime'].dt.to_period('M').astype(str)
monthly = filtered_df.groupby("month")["amount"].sum().sort_index()
st.bar_chart(monthly)

# --- WEEKLY SPENDING --- #
st.subheader("📆 Weekly Spending")
filtered_df['week'] = filtered_df['datetime'].dt.isocalendar().week
weekly = filtered_df.groupby("week")["amount"].sum().sort_index()
st.line_chart(weekly)

# --- DAILY HEATMAP --- #
st.subheader("🕒 Daily Spending Heatmap")
heatmap_data = filtered_df.copy()
heatmap_data['date'] = heatmap_data['datetime'].dt.date
heatmap = heatmap_data.groupby(['date'])['amount'].sum().reset_index()
heatmap['date'] = pd.to_datetime(heatmap['date'])
fig, ax = plt.subplots(figsize=(12, 4))
sns.lineplot(x='date', y='amount', data=heatmap, ax=ax)
ax.set_title("Daily Spending Over Time")
st.pyplot(fig)

# --- CATEGORY SPENDING --- #
st.subheader("📂 Spending by Category")
cat_data = filtered_df.groupby("category")["amount"].sum().sort_values(ascending=False)
st.bar_chart(cat_data)

# --- TIME OF DAY SPENDING --- #
st.subheader("⏰ Spending by Time of Day")
filtered_df['hour'] = filtered_df['datetime'].dt.hour
hourly = filtered_df.groupby('hour')['amount'].sum()
st.line_chart(hourly)

# --- CHATBOT --- #
st.subheader("💬 Ask Your Assistant")
user_input = st.chat_input("Talk to your finance assistant (e.g., 'How much did I spend on shopping in Feb 2024?')")
if user_input:
    response = chat_with_bot(user_input, filtered_df)
    st.success(response)

# --- OPTIONAL ENHANCEMENTS SECTION --- #
with st.expander("🛠 Optional Enhancements You Can Add"):
    st.markdown(""" 
    | Feature | Description |
    |--------|-------------|
    | 🔐 *Login System* | Secure access with username/password using hashed passwords |
    | 📥 *CSV Upload* | Upload your *own bank statements* in .csv format and view custom insights |
    | 🧠 *Smarter Chatbot* | Use *OpenAI/GPT* to answer complex queries like "What were my top 3 unnecessary expenses last month?" |
    | 🎯 *Budget Goals* | Set your own *monthly budget* and track progress visually |
    | 🏆 *Gamified Nudges* | Earn fun *badges/achievements* when you hit savings goals |
    | 📤 *Export Reports* | Export your data and insights to *PDF or Excel* format |
    """)
    st.info("💡 Let me know which one you want to build next and I’ll guide you step by step!")
