"""Predict the median house value of a California district.

    python predict.py --income 8.3 --lat 37.88 --lon -122.23 --age 41 --rooms 880 \
        --bedrooms 129 --population 322 --households 126 --ocean "NEAR BAY"

income is median household income in tens of thousands of US dollars (8.3 = $83,000), 1990 values.
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from housing import CATEGORICAL, NUMERIC, add_features  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
for a, d in [("income", 3.87), ("lat", 35.63), ("lon", -119.57), ("age", 29), ("rooms", 2636),
             ("bedrooms", 538), ("population", 1425), ("households", 500)]:
    ap.add_argument(f"--{a}", type=float, default=d)
ap.add_argument("--ocean", default="<1H OCEAN", choices=["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])
a = ap.parse_args()

row = pd.DataFrame([{"longitude": a.lon, "latitude": a.lat, "housing_median_age": a.age, "total_rooms": a.rooms,
                     "total_bedrooms": a.bedrooms, "population": a.population, "households": a.households,
                     "median_income": a.income, "ocean_proximity": a.ocean}])
model = pickle.load(open(Path(__file__).parent / "models" / "house_price_model.pkl", "rb"))
value = model.predict(add_features(row)[NUMERIC + CATEGORICAL])[0]
print(f"Predicted median house value: ${value:,.0f} (1990 dollars)")
