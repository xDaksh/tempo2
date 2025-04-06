import pandas as pd
import datetime
import streamlit as st

# 🚨 High Spending Alert
def high_spending_alert(df, threshold=2000):
    latest_spend = df["amount"].max()
    if latest_spend > threshold:
        return f"⚠️ High Spending Alert! You just spent ₹{latest_spend}. Review your budget!"
    return ""

# 📊 Predict Budget Overspending
def predict_overspending(df, budget):
    if df.empty:
        return ""
    
    avg_daily_spend = df.groupby(df['datetime'].dt.date)['amount'].sum().mean()
    days_left = (df['datetime'].max().date().replace(day=28) - datetime.date.today()).days
    projected_spend = avg_daily_spend * days_left + df["amount"].sum()

    if projected_spend > budget:
        return f"📉 Warning! You may exceed your budget by ₹{projected_spend - budget:.0f}. Adjust spending!"
    return "✅ Your spending is on track!"

# 🔄 Smart Category-Based Budget Adjustments
def adjust_category_budgets(df, category_budgets):
    warnings = []
    df['month'] = df['datetime'].dt.to_period('M')
    monthly_df = df[df['month'] == pd.Timestamp.now().to_period('M')]
    
    cat_spending = monthly_df.groupby("category")["amount"].sum()
    
    for category, spent in cat_spending.items():
        budget = category_budgets.get(category, None)
        if budget and spent > budget * 0.8:
            warnings.append(f"🔄 **{category}** spending is at {spent:.0f}/₹{budget}! Consider adjusting.")
    return warnings

# 🎯 Gamified Savings Streaks
def savings_streak_nudges(df, budget):
    total_saved = max(budget - df["amount"].sum(), 0)
    if total_saved >= 1000:
        return f"🔥 Streak Alert! You've saved ₹{total_saved}! Push to ₹1500 next week!"
    return ""

# 📆 Daily/Weekly Spending Summary
def spending_summary(df, period="daily"):
    today = datetime.date.today()
    
    if period == "daily":
        filtered_df = df[df["datetime"].dt.date == today]
        period_text = "Today"
    else:
        start_week = today - datetime.timedelta(days=today.weekday())  # Monday start
        filtered_df = df[df["datetime"].dt.date >= start_week]
        period_text = "This week"

    total_spent = filtered_df["amount"].sum()
    
    return f"📅 **{period_text}'s Summary:** You spent ₹{total_spent:.0f}."

# 📌 Compile All Nudges
def get_all_nudges(df, budget, category_budgets):
    nudges = []
    
    # 🚨 High spending alert
    high_spend = high_spending_alert(df)
    if high_spend:
        nudges.append(high_spend)
    
    # 📊 Budget overspending prediction
    overspending = predict_overspending(df, budget)
    if overspending:
        nudges.append(overspending)

    # 🔄 Category budget warnings
    nudges.extend(adjust_category_budgets(df, category_budgets))

    # 🎯 Gamified savings streaks
    streaks = savings_streak_nudges(df, budget)
    if streaks:
        nudges.append(streaks)

    # 📆 Daily & Weekly summary
    nudges.append(spending_summary(df, "daily"))
    nudges.append(spending_summary(df, "weekly"))

    return nudges

# 🚀 Example Usage
df = pd.DataFrame({
    "datetime": pd.date_range(start="2025-04-01", periods=10, freq="D"),
    "amount": [500, 1500, 2000, 300, 700, 2500, 1200, 1800, 600, 400],
    "category": ["Food", "Shopping", "Entertainment", "Groceries", "Bills", "Food", "Shopping", "Transport", "Bills", "Food"]
})

budget = 10000
category_budgets = {"Food": 3000, "Shopping": 2500, "Entertainment": 2000, "Bills": 2000}

nudges = get_all_nudges(df, budget, category_budgets)

# 📢 Display Nudges
for nudge in nudges:
    st.write(nudge)
