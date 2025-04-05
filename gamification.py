
import pandas as pd
from datetime import datetime, timedelta

class BadgeSystem:
    def __init__(self):
        self.badges = {
            'bronze': [
                {'threshold': 100, 'icon': '🥉', 'name': 'Bronze Saver I'},
                {'threshold': 200, 'icon': '🥉', 'name': 'Bronze Saver II'},
                {'threshold': 300, 'icon': '🥉', 'name': 'Bronze Saver III'}
            ],
            'silver': [
                {'threshold': 500, 'icon': '🥈', 'name': 'Silver Saver I'},
                {'threshold': 750, 'icon': '🥈', 'name': 'Silver Saver II'},
                {'threshold': 1000, 'icon': '🥈', 'name': 'Silver Saver III'}
            ],
            'gold': [
                {'threshold': 1500, 'icon': '🥇', 'name': 'Gold Saver I'},
                {'threshold': 2000, 'icon': '🥇', 'name': 'Gold Saver II'},
                {'threshold': 2500, 'icon': '🥇', 'name': 'Gold Saver III'}
            ],
            'diamond': [
                {'threshold': 5000, 'icon': '💎', 'name': 'Diamond Saver'}
            ]
        }

    def calculate_badges(self, total_saved):
        earned_badges = []
        for tier in ['bronze', 'silver', 'gold', 'diamond']:
            for badge in self.badges[tier]:
                if total_saved >= badge['threshold']:
                    earned_badges.append(f"{badge['icon']} {badge['name']}")
        return earned_badges

    def get_next_badge(self, total_saved):
        next_badge = None
        next_threshold = float('inf')
        
        for tier in ['bronze', 'silver', 'gold', 'diamond']:
            for badge in self.badges[tier]:
                if badge['threshold'] > total_saved and badge['threshold'] < next_threshold:
                    next_badge = badge
                    next_threshold = badge['threshold']
        
        return next_badge if next_badge else None

    def get_progress(self, total_saved):
        next_badge = self.get_next_badge(total_saved)
        if next_badge:
            # Find previous threshold
            prev_threshold = 0
            for tier in ['bronze', 'silver', 'gold', 'diamond']:
                for badge in self.badges[tier]:
                    if badge['threshold'] < next_badge['threshold'] and badge['threshold'] > prev_threshold:
                        prev_threshold = badge['threshold']
            
            progress = (total_saved - prev_threshold) / (next_badge['threshold'] - prev_threshold)
            return progress, next_badge
        return 1.0, None
