import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import razorpay
import time
import threading
import openai
import sqlite3
import bcrypt
from gamification import BadgeSystem
from popup import show_completion_popup

# Set page config before any other st calls
st.set_page_config(page_title="AI Finance Assistant", layout="wide")

# Custom CSS for splash screen
splash_css = """
    <style>
    [data-testid="stImage"] img {
        width: 100vw !important;
        height: 100vh !important;
        object-fit: cover;
    }
    </style>
"""

# Initialize session states
if "splash_shown" not in st.session_state:
    st.session_state["splash_shown"] = False
if "budget" not in st.session_state:
    st.session_state["budget"] = 20000
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Database functions
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)
    conn.commit()
    conn.close()

def register_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def authenticate_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode(), user[0]):
        return True
    return False

# Initialize DB
init_db()

# Show splash screen if not already shown
if not st.session_state["splash_shown"]:
    st.markdown(splash_css, unsafe_allow_html=True)
    splash = st.empty()
    splash.image("splash.png", width=800)

    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.03)
        progress_bar.progress(i + 1)

    splash.empty()
    progress_bar.empty()
    st.session_state["splash_shown"] = True

def login_page():
    st.title("🔐 Welcome to AI Finance Assistant")
    st.markdown("""
    <style>
    .stTextInput>div>div>input { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

    choice = st.radio("Select an option", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if choice == "Login":
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success("✅ Login Successful! Redirecting...")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    else:
        if st.button("Register"):
            if register_user(username, password):
                st.success("✅ Registration Successful! Please log in.")
            else:
                st.error("⚠ Username already exists")

# Authentication Check
if not st.session_state["authenticated"]:
    login_page()
    st.stop()

# Load and process data
@st.cache_data
def load_data():
    df = pd.read_csv("mock_transactions_detailed.csv")
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

df = load_data()

# Sidebar setup
st.sidebar.title(f"👋 Welcome, {st.session_state['username']}")
st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"authenticated": False}))

# Main application layout
st.title("💰 AI Finance Assistant Dashboard")

# Budget settings and tracking
monthly_budget = st.session_state["budget"]
df_current_month = df[df['datetime'].dt.strftime('%Y-%m') == datetime.now().strftime('%Y-%m')]
total_spent = df_current_month['amount'].sum()
remaining_budget = monthly_budget - total_spent

# Display budget metrics
col1, col2, col3 = st.columns(3)
col1.metric("Monthly Budget", f"₹{monthly_budget:,.2f}")
col2.metric("Total Spent", f"₹{total_spent:,.2f}")
col3.metric("Remaining", f"₹{remaining_budget:,.2f}")

# Category-wise spending
st.subheader("📊 Category-wise Spending")
category_spending = df_current_month.groupby('category')['amount'].sum()
st.bar_chart(category_spending)

# Time series of spending
st.subheader("📈 Daily Spending Trend")
daily_spending = df_current_month.groupby(df_current_month['datetime'].dt.date)['amount'].sum()
st.line_chart(daily_spending)


# -------------------------------
# Helper Functions
# -------------------------------

def pre_populate_badges_for_previous_months(df, budget):
    months = df['datetime'].dt.to_period('M').unique()
    today = datetime.today().date()
    badges_earned_for_previous_months = []
    for month in months:
        if month < pd.to_datetime(today.replace(day=1)).to_period('M'):
            monthly_data = df[df['datetime'].dt.to_period('M') == month]
            total_spent = monthly_data["amount"].sum()
            total_saved = budget - total_spent
            badge_system = BadgeSystem()
            earned_badges = badge_system.calculate_badges(total_saved)
            for badge in earned_badges:
                badges_earned_for_previous_months.append(f"🏅 {badge['name']} for {month}")
    return badges_earned_for_previous_months

def get_gamified_nudges(df, budget):
    badges_earned = []
    today = datetime.today()
    tomorrow = today + timedelta(days=1)

    if tomorrow.month != today.month:
        expenditure = df["amount"].sum()
        total_saved = budget - expenditure

        if "popup_shown_for" not in st.session_state:
            st.session_state["popup_shown_for"] = ""

        if st.session_state["popup_shown_for"] != today.strftime("%Y-%m"):
            badge_system = BadgeSystem()
            earned_badges = badge_system.calculate_badges(total_saved)

            show_completion_popup(f"🎉 Month End Summary: You saved ₹{total_saved:.0f}!", duration=4)

            for badge in earned_badges:
                show_completion_popup(f"🏅 {badge['name']}", image_path=badge['image'], duration=3)
                badges_earned.append(f"🏅 {badge['name']}")

            progress, next_badge = badge_system.get_progress(total_saved)
            if next_badge:
                show_completion_popup(
                    f"Next badge: {next_badge['name']} ({progress*100:.1f}% complete)",
                    image_path=next_badge['image'],
                    duration=3
                )

            st.session_state["popup_shown_for"] = today.strftime("%Y-%m")

    return badges_earned

def get_category_warnings(df, category_budgets):
    warnings = []
    for cat in df['category'].unique():
        spent = df[df['category'] == cat]['amount'].sum()
        if cat in category_budgets and spent > category_budgets[cat]:
            warnings.append(f"⚠ {cat}: Spent ₹{spent:.0f}, which exceeds your category budget of ₹{category_budgets[cat]:.0f}")
    return warnings

def chat_with_bot(query, df):
    if not st.session_state.get("openai_api_key"):
        return "⚠ Please enter your OpenAI API Key in the sidebar."

    openai.api_key = st.session_state["openai_api_key"]

    recent_data = df.tail(10)[["datetime", "amount", "category", "type"]].to_string(index=False)

    prompt = f"""
    You are a smart personal finance assistant. Help me analyze and understand my spending.

    Here are my recent transactions:
    {recent_data}

    Now answer this question based on the data above:

    {query}

    Be brief, helpful, and clear.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"🚨 Error: {e}"

# Sidebar Controls
st.sidebar.title("🔍 Navigation")
st.sidebar.divider()

min_date = df["datetime"].min().date()
max_date = df["datetime"].max().date()
date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
if len(date_range) != 2:
    st.stop()

types = df["type"].unique()
selected_type = st.sidebar.multiselect("Transaction Type", types, default=list(types))

categories = df["category"].unique()
selected_category = st.sidebar.multiselect("Categories", categories, default=list(categories))

st.sidebar.markdown("### 💸 Budget Settings")
if "budget" not in st.session_state:
    st.session_state.budget = 50000
budget = st.sidebar.number_input("Set Monthly Budget", value=st.session_state.budget, step=1000)
st.session_state.budget = budget

st.sidebar.markdown("### 📂 Category Budgets")
category_budgets = {}
for cat in categories:
    cat_budget = st.sidebar.number_input(f"{cat} Budget", min_value=0, value=5000, key=f"{cat}_budget")
    category_budgets[cat] = cat_budget


st.sidebar.markdown("### 💳 Razorpay Login")
if "razorpay_key" not in st.session_state:
    st.session_state["razorpay_key"] = ""
    st.session_state["razorpay_secret"] = ""

api_key = st.sidebar.text_input("API Key", type="password", value=st.session_state["razorpay_key"])
api_secret = st.sidebar.text_input("API Secret", type="password", value=st.session_state["razorpay_secret"])

if st.sidebar.button("🔓 Authenticate Razorpay", type="primary"):
    if api_key and api_secret:
        st.session_state["razorpay_key"] = api_key
        st.session_state["razorpay_secret"] = api_secret
        st.sidebar.success("✅ Authentication Successful!")
    else:
        st.sidebar.error("⚠ Please enter both API Key and Secret.")

st.sidebar.markdown("### 🤖 OpenAI API Key (for Chatbot)")
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

api_key_input = st.sidebar.text_input("Enter OpenAI API Key", type="password", value=st.session_state["openai_api_key"])
if api_key_input:
    st.session_state["openai_api_key"] = api_key_input
    openai.api_key = api_key_input

page_options = [
    "🏠 Dashboard",
    "📊 Expense Forecasting",
    "🔍 Category-wise Expense Forecasting",
    "📅 Monthly Spending",
    "📆 Weekly Spending",
    "📂 Spending by Category",
    "🏆 Achievement Nudges",
    "⚠ Budget Warnings",
    "💬 AI Chatbot",
    "💳 Razorpay Tracking"
]

selected_page = st.sidebar.radio("Go to section:", page_options)

filtered_df = df[
    (df["type"].isin(selected_type)) &
    (df["category"].isin(selected_category)) &
    (df["datetime"].dt.date >= date_range[0]) &
    (df["datetime"].dt.date <= date_range[1])
]

if selected_page == "🏠 Dashboard":
    st.title("💰 AI Finance Assistant Dashboard")
    st.subheader("📈 Quick Summary (This Month)")

    df['month'] = df['datetime'].dt.to_period('M')
    monthly_expenses = df.groupby('month')['amount'].sum().reset_index()

    current_month_expenses = monthly_expenses[monthly_expenses['month'] == pd.to_datetime(pd.Timestamp.today().replace(day=1)).to_period('M')]

    if not current_month_expenses.empty:
        total_spent_monthly = current_month_expenses['amount'].sum()
    else:
        total_spent_monthly = 0

    current_month_transactions = df[df['datetime'].dt.to_period('M') == pd.to_datetime(pd.Timestamp.today().replace(day=1)).to_period('M')]
    num_transactions_monthly = len(current_month_transactions)

    st.write(f"Total Spent This Month: ₹{total_spent_monthly:,.2f}")
    st.write(f"Number of Transactions This Month: {num_transactions_monthly}")

    st.subheader("📂 Spending by Category (This Month)")

    # Filter the data to include only the current month's transactions
    category_expenses = current_month_transactions.groupby('category')['amount'].sum().sort_values(ascending=False)
    st.bar_chart(category_expenses)

    show_yearly_button = st.button("Show Yearly Data")
    if show_yearly_button:
        st.subheader("📊 Yearly Summary")
        df['year'] = df['datetime'].dt.year
        yearly_expenses = df.groupby('year')['amount'].sum().reset_index()
        st.bar_chart(yearly_expenses.set_index('year')['amount'])

elif selected_page == "📊 Expense Forecasting":
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

elif selected_page == "🔍 Category-wise Expense Forecasting":
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
        forecast_msg = f"📌 {cat}: Forecasted ₹{forecast:.0f} / Budget ₹{cat_budget}"
        if forecast > cat_budget:
            st.warning(f"🚨 {forecast_msg} — Likely to overspend!")
        else:
            st.info(f"✅ {forecast_msg} — Looks safe.")

elif selected_page == "📅 Monthly Spending":
    st.subheader("📅 Monthly Spending")
    filtered_df['month'] = filtered_df['datetime'].dt.to_period('M').astype(str)
    monthly = filtered_df.groupby("month")["amount"].sum().sort_index()
    st.bar_chart(monthly)

elif selected_page == "📆 Weekly Spending":
    st.subheader("📆 Weekly Spending")
    filtered_df['week'] = filtered_df['datetime'].dt.isocalendar().week
    weekly = filtered_df.groupby("week")["amount"].sum().sort_index()
    st.line_chart(weekly)

elif selected_page == "📂 Spending by Category":
    st.subheader("📂 Spending by Category")
    cat_data = filtered_df.groupby("category")["amount"].sum().sort_values(ascending=False)
    st.bar_chart(cat_data)

elif selected_page == "🏆 Achievement Nudges":
    st.subheader("🏆 Achievement Nudges")
    badges_from_previous_months = pre_populate_badges_for_previous_months(df, st.session_state.budget)
    for badge in badges_from_previous_months:
        st.success(badge)
    df_this_month = filtered_df[filtered_df["datetime"].dt.to_period('M').astype(str) == datetime.now().strftime('%Y-%m')]
    badges = get_gamified_nudges(df_this_month, budget)
    for badge in badges:
        st.success(badge)

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

    savings = budget - df_this_month["amount"].sum()
    badge = get_savings_badge(savings)
    if badge:
        st.success(f"🏆 {badge}")

elif selected_page == "⚠ Budget Warnings":
    st.subheader("⚠ Category Budget Warnings")
    df_this_month = filtered_df[filtered_df["datetime"].dt.to_period('M').astype(str) == datetime.now().strftime('%Y-%m')]
    category_warnings = get_category_warnings(df_this_month, category_budgets)
    for warning in category_warnings:
        st.warning(warning)

elif selected_page == "💬 AI Chatbot":
    st.subheader("💬 Ask Your Assistant")
    st.write("Ask me questions about your financial behavior")

    user_input = st.chat_input("Talk to your financal assistant Penny")
    if user_input:
        response = chat_with_bot(user_input, filtered_df)
        st.success(response)

elif selected_page == "💳 Razorpay Tracking":
    st.subheader("💳 Live Razorpay Transactions")

    def fetch_latest_payments():
        if not st.session_state["razorpay_key"] or not st.session_state["razorpay_secret"]:
            st.warning("⚠ Please authenticate with Razorpay API.")
            return []

        try:
            client = razorpay.Client(auth=(st.session_state["razorpay_key"], st.session_state["razorpay_secret"]))
            payments = client.payment.all({"count": 10})
            transactions = []
            for payment in payments['items']:
                transactions.append({
                    "id": payment["id"],
                    "amount": payment["amount"] / 100,
                    "method": payment.get("method", "Unknown").title(),
                    "status": payment["status"],
                    "created_at": pd.to_datetime(payment["created_at"], unit='s')
                })
            return transactions
        except Exception as e:
            st.error(f"Error: {e}")
            return []

    def start_realtime_tracking():
        seen_ids = set()
        st.session_state["tracking"] = True
        while st.session_state["tracking"]:
            transactions = fetch_latest_payments()
            new_rows = []
            for txn in transactions:
                if txn["id"] not in seen_ids and txn["status"] == "captured":
                    seen_ids.add(txn["id"])
                    new_rows.append(txn)

            if new_rows:
                new_df = pd.DataFrame(new_rows)
                try:
                    existing_df = pd.read_csv("razorpay_payments.csv")
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                except FileNotFoundError:
                    combined_df = new_df

                combined_df.to_csv("razorpay_payments.csv", index=False)
                st.success(f"✅ {len(new_rows)} new transactions added!")
            time.sleep(5)

    def start_tracking_thread():
        if not st.session_state.get("tracking", False):
            thread = threading.Thread(target=start_realtime_tracking, daemon=True)
            thread.start()
            st.success("✅ Real-time tracking started!")

    def stop_tracking():
        st.session_state["tracking"] = False
        st.warning("⚠ Tracking stopped!")

    col1, col2 = st.columns(2)
    if col1.button("▶ Start Tracking", use_container_width=True):
        start_tracking_thread()
    if col2.button("⏹ Stop Tracking", use_container_width=True):
        stop_tracking()

    if st.button("🔄 Refresh Transactions"):
        transactions = fetch_latest_payments()
        if transactions:
            for txn in transactions:
                status_color = {
                    "captured": "green",
                    "failed": "red",
                    "created": "orange"
                }.get(txn["status"], "gray")

                st.markdown(f"""
                    <div style="border-left: 4px solid {status_color}; padding: 10px; margin: 5px 0; background-color: #f9f9f9; border-radius: 6px;">
                        🆔 <b>{txn['id']}</b><br>
                        ₹{txn['amount']} | 🏦 {txn['method']} | <b style="color:{status_color}">{txn['status'].title()}</b><br>
                        📅 {txn['created_at']}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No transactions found.")
