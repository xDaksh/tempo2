import pandas as pd
from datetime import datetime, timedelta
from popup import show_completion_popup

class BadgeSystem:
    def __init__(self):
        self.badges = {
            'bronze': [
                {'threshold': 100, 'icon': '🥉', 'name': 'Bronze Saver I ', 'image': 'bronze 1.jpg'},
                {'threshold': 200, 'icon': '🥉', 'name': 'Bronze Saver II', 'image': 'bronze 2.jpg'},
                {'threshold': 300, 'icon': '🥉', 'name': 'Bronze Saver III', 'image': 'bronze 3.jpg'}
            ],
            'silver': [
                {'threshold': 500, 'icon': '🥈', 'name': 'Silver Saver I', 'image': 'silver 1.jpg'},
                {'threshold': 750, 'icon': '🥈', 'name': 'Silver Saver II', 'image': 'silver 2.jpg'},
                {'threshold': 1000, 'icon': '🥈', 'name': 'Silver Saver III', 'image': 'silver 3.jpg'}
            ],
            'gold': [
                {'threshold': 1500, 'icon': '🥇', 'name': 'Gold Saver I', 'image': 'gold 1.jpg'},
                {'threshold': 2000, 'icon': '🥇', 'name': 'Gold Saver II', 'image': 'gold 2.jpg'},
                {'threshold': 2500, 'icon': '🥇', 'name': 'Gold Saver III', 'image': 'gold 3.jpg'}
            ],
            'diamond': [
                {'threshold': 5000, 'icon': '💎', 'name': 'Diamond Saver', 'image': 'diamond .jpg'}
            ]
        }

    def calculate_badges(self, total_saved):
        earned_badges = []
        previous_badges = getattr(self, '_previous_badges', set())
        current_badges = set()

        for tier in ['bronze', 'silver', 'gold', 'diamond']:
            for badge in self.badges[tier]:
                if total_saved >= badge['threshold']:
                    badge_data = {
                        'name': f"{badge['icon']} {badge['name']}",
                        'image': badge['image']
                    }
                    earned_badges.append(badge_data)
                    current_badges.add(badge['name'])

                    # Show popup only for newly earned badges
                    if badge['name'] not in previous_badges:
                        show_completion_popup(badge['image'], duration=3)

        self._previous_badges = current_badges
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