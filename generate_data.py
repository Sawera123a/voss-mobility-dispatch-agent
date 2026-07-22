"""
Generates synthetic ride-demand data for the Voss Mobility

Simulates 6 zones across a city, over 30 days, with hourly demand
and vehicle-availability counts. Demand patterns include realistic
rush-hour peaks (morning + evening) and weekend differences.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Config
NUM_DAYS = 30
ZONES = ["Zone_A", "Zone_B", "Zone_C", "Zone_D", "Zone_E", "Zone_F"]
START_DATE = datetime(2026, 6, 1)
np.random.seed(42)  # reproducible results

rows = []

for day_offset in range(NUM_DAYS):
    current_date = START_DATE + timedelta(days=day_offset)
    day_of_week = current_date.weekday()  
    is_weekend = day_of_week >= 5

    for hour in range(24):
        for zone in ZONES:
            # Base demand varies by zone (some zones are busier overall)
            zone_base = {
                "Zone_A": 25, "Zone_B": 15, "Zone_C": 30,
                "Zone_D": 10, "Zone_E": 20, "Zone_F": 12,
            }[zone]

            # Rush hour boost (8-9 AM, 5-7 PM) on weekdays
            rush_boost = 0
            if not is_weekend and hour in [8, 9, 17, 18, 19]:
                rush_boost = np.random.randint(15, 35)

            # Weekend night boost (nightlife demand, 9 PM - 1 AM)
            weekend_night_boost = 0
            if is_weekend and (hour >= 21 or hour <= 1):
                weekend_night_boost = np.random.randint(10, 25)

            # Random noise
            noise = np.random.randint(-5, 6)

            demand = max(0, zone_base + rush_boost + weekend_night_boost + noise)

            # Available vehicles: loosely tied to demand but imperfect
            # (this imperfection is what causes the business problem)
            available_vehicles = max(
                0, int(demand * np.random.uniform(0.4, 0.9)) + np.random.randint(-3, 4)
            )

            rows.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "day_of_week": day_of_week,
                "is_weekend": int(is_weekend),
                "hour": hour,
                "zone": zone,
                "ride_demand": demand,
                "available_vehicles": available_vehicles,
            })

df = pd.DataFrame(rows)
df.to_csv("data/ride_demand.csv", index=False)

print(f"Generated {len(df)} rows across {NUM_DAYS} days and {len(ZONES)} zones.")
print("Saved to data/ride_demand.csv")
print("\nPreview:")
print(df.head(10))
