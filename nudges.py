import pandas as pd
import numpy as np
import datetime

# AI-Powered Spending Predictions
def predict_overspending(df, budget):
    if df.empty:
        return "📭 No transactions found for prediction."
    
    avg_daily_spend = df.groupby(df['datetime'].dt.date)['amount'].sum().mean()
    days_left = (df['datetime'].max().date().replace(day=28) - datetime.date.today()).days
    projected_spend = avg_daily_spend * days_left + df["amount"].sum()
    
    if projected_spend > budget:
        return f"📊 Based on your spending trend, you may exceed your budget by ₹{projected_spend - budget:.0f}! Consider adjusting your expenses."
    return "✅ Your spending is on track!"

# Contextual Spending Nudges
def spending_trends_nudges(df):
    nudges = []
    if df.empty:
        return nudges
    
    df['day_of_week'] = df['datetime'].dt.day_name()
    weekend_spend = df[df['day_of_week'].isin(["Saturday", "Sunday"])]
    weekday_spend = df[~df['day_of_week'].isin(["Saturday", "Sunday"])]
    
    if not weekend_spend.empty and not weekday_spend.empty:
        if weekend_spend["amount"].sum() > weekday_spend["amount"].sum() * 1.3:
            nudges.append("🚀 Your weekend spends are 30% higher than weekdays! Try planning meals at home to save.")
    return nudges

# Personalized Goal-Setting & Streaks
def savings_streak_nudges(df, budget):
    nudges = []
    total_saved = max(budget - df["amount"].sum(), 0)
    
    if total_saved >= 500:
        nudges.append(f"🌟 You've saved ₹{total_saved}! Try pushing it to ₹750 next week!")
    return nudges

# Custom Achievements & Leaderboard
def achievement_badges(df, budget):
    total_saved = max(budget - df["amount"].sum(), 0)
    badges = []
    
    if total_saved >= 2000:
        badges.append("💎 You earned the **Diamond** badge for saving ₹2000+!")
    elif total_saved >= 1000:
        badges.append("🏆 Platinum badge unlocked! Keep going!")
    elif total_saved >= 500:
        badges.append("🥇 Gold badge achieved! You're on fire!")
    return badges

# Streak-Based Cashback (Mocked Rewards)
def cashback_rewards(df, budget):
    total_saved = max(budget - df["amount"].sum(), 0)
    if total_saved >= 1000:
        return "🎁 You unlocked a mock ₹50 discount for saving ₹1000+ this month!"
    return ""

# Instant Notifications on High Spending
def high_spending_alert(df, threshold=2000):
    if df["amount"].max() > threshold:
        return f"⚠️ You just spent ₹{df['amount'].max()}! Consider reviewing your expenses."
    return ""

# Smart Category-Based Budget Adjustments
def adjust_category_budgets(df, category_budgets):
    warnings = []
    df['month'] = df['datetime'].dt.to_period('M')
    monthly_df = df[df['month'] == pd.Timestamp.now().to_period('M')]
    
    cat_spending = monthly_df.groupby("category")["amount"].sum()
    
    for category, spent in cat_spending.items():
        budget = category_budgets.get(category, None)
        if budget and spent > budget * 0.8:
            warnings.append(f"🔄 Consider shifting funds from other categories as **{category}** is nearing its budget!")
    return warnings

# 📌 NEW FUNCTION: Get Category-Specific Warnings
def get_category_warnings(df, category_budgets):
    """
    Generates warnings if spending in any category exceeds the budget.
    """
    warnings = []
    if df.empty or not category_budgets:
        return warnings

    category_totals = df.groupby("category")["amount"].sum()

    for category, spent in category_totals.items():
        budget = category_budgets.get(category, 0)
        if spent > budget:
            warnings.append(f"⚠ Over budget in {category}: Spent ₹{spent:.0f} / Budget ₹{budget}")

    return warnings

# Main function to get all nudges
def get_gamified_nudges(df, budget, category_budgets=None):
    nudges = []
    
    nudges.extend(spending_trends_nudges(df))
    nudges.append(predict_overspending(df, budget))
    nudges.extend(savings_streak_nudges(df, budget))
    nudges.extend(achievement_badges(df, budget))
    cashback_msg = cashback_rewards(df, budget)
    if cashback_msg:
        nudges.append(cashback_msg)
    
    high_spend_msg = high_spending_alert(df)
    if high_spend_msg:
        nudges.append(high_spend_msg)
    
    category_warnings = get_category_warnings(df, category_budgets)
    nudges.extend(category_warnings)
    
    return nudges

