import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
s2021 = pd.read_csv(r"C:\Users\User\Downloads\Premier League Machine Learning Project\2021-22 Prem stats.csv")
s2022 = pd.read_csv(r"C:\Users\User\Downloads\Premier League Machine Learning Project\2022-23 Prem stats.csv")
s2023 = pd.read_csv(r"C:\Users\User\Downloads\Premier League Machine Learning Project\2023-24 Prem stats.csv")
s2024 = pd.read_csv(r"C:\Users\User\Downloads\Premier League Machine Learning Project\2024-25 Prem stats.csv")
s2025 = pd.read_csv(r"C:\Users\User\Downloads\Premier League Machine Learning Project\2025-26 Prem stats.csv")
All_seasons = pd.concat([s2021,s2022,s2023,s2024,s2025])
key_stats = All_seasons[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]] ## this part shortens the table to the 6 elements I care about most, data, home team, away team, home goals, away goals and full time result
key_stats["Date"] = pd.to_datetime(key_stats["Date"], dayfirst=True)
key_stats = key_stats.sort_values("Date").reset_index(drop=True)
print(key_stats["FTR"].value_counts(normalize=True))
key_stats["match_id"] = key_stats.index

home = pd.DataFrame({
    "match_id": key_stats["match_id"], "Date": key_stats["Date"], "Team": key_stats["HomeTeam"],
    "GF": key_stats["FTHG"], "GA": key_stats["FTAG"], "is_home": True})
away = pd.DataFrame({
    "match_id": key_stats["match_id"], "Date": key_stats["Date"], "Team": key_stats["AwayTeam"],
    "GF": key_stats["FTAG"], "GA": key_stats["FTHG"], "is_home": False})
teams = pd.concat([home, away], ignore_index=True)

teams["Points"] = np.where(teams["GF"] > teams["GA"], 3,
                  np.where(teams["GF"] == teams["GA"], 1, 0))

teams = teams.sort_values(["Team", "Date"]).reset_index(drop=True)

def form(column):
    return teams.groupby("Team")[column].transform(
        lambda s: s.rolling(5).mean().shift(1)) ## finds average of stat over the last 3 matches, shift pushes down the values as another match is added

teams["form_pts"] = form("Points")
teams["form_gf"] = form("GF")
teams["form_ga"] = form("GA")  ## repeats the form functions for useful stats, goals for, goals against and points

cols = ["form_pts", "form_gf", "form_ga"]
home_form = teams[teams["is_home"]].set_index("match_id")[cols].add_prefix("home_") ## only keeps the matches that were played at home
away_form = teams[~teams["is_home"]].set_index("match_id")[cols].add_prefix("away_") ## only keeps the matches that aren't at home

data = key_stats.set_index("match_id").join(home_form).join(away_form)
data = data.dropna().reset_index(drop=True)
print(data.shape)


features = ["home_form_pts", "home_form_gf", "home_form_ga",
            "away_form_pts", "away_form_gf", "away_form_ga"]


train = data[data["Date"] < "2025-08-01"]
test = data[data["Date"] >= "2025-08-01"]



print("Baseline:", (test["FTR"] == "H").mean())


model1 = LogisticRegression(max_iter=1000)
model1.fit(train[features], train["FTR"])
print("Logistic regression:", model1.score(test[features], test["FTR"]))


model2 = RandomForestClassifier(max_depth=5, random_state=42)
model2.fit(train[features], train["FTR"])
print("Random forest:", model2.score(test[features], test["FTR"]))

def test_prediction(home, away):
    match = test[(test["HomeTeam"] == home) & (test["AwayTeam"] == away)]
    if match.empty:
        print(f"No {home} vs {away} match found in the test data.")
        return
    real_result = match["FTR"].iloc[0]
    print(match[["HomeTeam","AwayTeam","Date","FTR"]])
    if real_result == "H":
         print("result was a Home win")
    elif real_result == "A":
         print("result was an Away win")
    else:
         print("result was a Draw")
    if model1.predict(match[features])[0] == "H":
        print("Model 1 predicts Home win")
    elif model1.predict(match[features])[0] == "A":
        print("Model 1 predicts Away win")
    else:
        print("Model 1 predicts Draw")
    if model2.predict(match[features])[0] == "H":
            print("Model 2 predicts Home win")
    elif model2.predict(match[features])[0] == "A":
         print("Model 2 predicts Away win")
    else:
        print("Model 2 predicts Draw")
print("\nLogistic Regression predictions:")
print(pd.Series(model1.predict(test[features])).value_counts())

print("\nRandom Forest predictions:")
print(pd.Series(model2.predict(test[features])).value_counts())

test_prediction("Liverpool", "Man City")
test_prediction("Arsenal", "Chelsea")
test_prediction("Wolves", "Man City")