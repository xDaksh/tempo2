import streamlit as st
import razorpay
import pandas as pd
import time
import threading
from datetime import datetime

# --- UI Styling ---
st.markdown("""
    <style>
        .razorpay-title {
            font-size: 22px;
            font-weight: bold;
            color: #0078D7;
            text-align: center;
        }
        .api-box {
            border: 2px solid #0078D7;
            border-radius: 10px;
            padding: 10px;
            background-color: #F0F8FF;
        }
        .txn-success { color: green; font-weight: bold; }
        .txn-failed { color: red; font-weight: bold; }
        .txn-pending { color: orange; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar for API Authentication ---
st.sidebar.markdown('<p class="razorpay-title">🔑 Razorpay Login</p>', unsafe_allow_html=True)
if "razorpay_key" not in st.session_state:
    st.session_state["razorpay_key"] = ""
    st.session_state["razorpay_secret"] = ""

with st.sidebar:
    st.markdown('<div class="api-box">Enter your Razorpay API Details:</div>', unsafe_allow_html=True)
    api_key = st.text_input("API Key", type="password", value=st.session_state["razorpay_key"])
    api_secret = st.text_input("API Secret", type="password", value=st.session_state["razorpay_secret"])

    if st.button("🔓 Authenticate"):
        if api_key and api_secret:
            st.session_state["razorpay_key"] = api_key
            st.session_state["razorpay_secret"] = api_secret
            st.sidebar.success("✅ Authentication Successful!")
        else:
            st.sidebar.error("⚠ Please enter both API Key and Secret!")

# --- Function to Fetch Transactions ---
def fetch_latest_payments():
    """Fetches latest transactions from Razorpay API."""
    if not st.session_state["razorpay_key"] or not st.session_state["razorpay_secret"]:
        st.warning("⚠ Please login with your Razorpay API details first.")
        return []

    try:
        client = razorpay.Client(auth=(st.session_state["razorpay_key"], st.session_state["razorpay_secret"]))
        payments = client.payment.all({"count": 10})

        transactions = []
        for payment in payments['items']:
            transactions.append({
                "id": payment["id"],
                "amount": payment["amount"] / 100,  # Convert to ₹
                "method": payment.get("method", "Unknown").title(),
                "status": payment["status"],
                "created_at": pd.to_datetime(payment["created_at"], unit='s')
            })
        return transactions

    except Exception as e:
        st.error(f"⚠ Error fetching transactions: {e}")
        return []

# --- Background Tracking Function ---
def start_realtime_tracking():
    """Continuously fetches new transactions every 5 seconds and updates CSV."""
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

        time.sleep(5)  # Fetch new transactions every 5 seconds

# --- Function to Start Tracking in Background ---
def start_tracking_thread():
    """Runs tracking in a background thread."""
    if not st.session_state.get("tracking", False):  # Avoid multiple threads
        thread = threading.Thread(target=start_realtime_tracking, daemon=True)
        thread.start()
        st.sidebar.success("✅ Real-time tracking started!")

# --- Function to Stop Tracking ---
def stop_tracking():
    """Stops real-time tracking."""
    st.session_state["tracking"] = False
    st.sidebar.warning("⚠ Tracking stopped!")

# --- Display Transactions ---
st.markdown('<p class="razorpay-title">💸 Live Razorpay Transactions</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Real-Time Tracking"):
        start_tracking_thread()

with col2:
    if st.button("⏹ Stop Tracking"):
        stop_tracking()

if st.button("🔄 Refresh Transactions"):
    transactions = fetch_latest_payments()

    if transactions:
        for txn in transactions:
            status_icon = {
                "captured": "✅ <span class='txn-success'>Success</span>",
                "failed": "❌ <span class='txn-failed'>Failed</span>",
                "created": "⏳ <span class='txn-pending'>Pending</span>"
            }.get(txn["status"], "🔍 Unknown")

            st.markdown(f"""
                <div style="border: 1px solid #0078D7; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
                    🆔 <b>{txn['id']}</b> | ₹{txn['amount']} | 🏦 {txn['method']} | {status_icon} | 📅 {txn['created_at']}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("🚫 No transactions found.")
