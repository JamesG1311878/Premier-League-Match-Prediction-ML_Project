import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

DATA = Path(__file__).parent / "data"
s2021 = pd.read_csv(DATA / "2021-22 Prem stats.csv")
s2022 = pd.read_csv(DATA / "2022-23 Prem stats.csv")
s2023 = pd.read_csv(DATA / "2023-24 Prem stats.csv")
s2024 = pd.read_csv(DATA / "2024-25 Prem stats.csv")
s2025 = pd.read_csv(DATA / "2025-26 Prem stats.csv")
All_seasons = pd.concat([s2021,s2022,s2023,s2024,s2025])
key_stats = All_seasons[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]] ## this part shortens the table to the 6 elements I care about most, data, home team, away team, home goals, away goals and full time result
key_stats["Date"] = pd.to_datetime(key_stats["Date"], dayfirst=True)
key_stats = key_stats.sort_values("Date").reset_index(drop=True) ## I needed to do some research for this part as I did not realise that the computer was interpreting the dates as text, so I found this code which can change the text into an actual date in UK format so they can be put in chronological order in the tables
print(key_stats["FTR"].value_counts(normalize=True)) ## looks at probabilities of a home win, away win and draw and outputs them
key_stats["match_id"] = key_stats.index

home = pd.DataFrame({
    "match_id": key_stats["match_id"], "Date": key_stats["Date"], "Team": key_stats["HomeTeam"],
    "GF": key_stats["FTHG"], "GA": key_stats["FTAG"], "is_home": True}) ## this table has data from the perspective of the home team, it will be joined on to the rest later
away = pd.DataFrame({
    "match_id": key_stats["match_id"], "Date": key_stats["Date"], "Team": key_stats["AwayTeam"],
    "GF": key_stats["FTAG"], "GA": key_stats["FTHG"], "is_home": False}) ## this table has data from the perspective of the away team, it will be joined on to the rest later
teams = pd.concat([home, away], ignore_index=True)

teams["Points"] = np.where(teams["GF"] > teams["GA"], 3,
                  np.where(teams["GF"] == teams["GA"], 1, 0))

teams = teams.sort_values(["Team", "Date"]).reset_index(drop=True)

def form(column):
    return teams.groupby("Team")[column].transform( ## spent a while without the .shift(1) and this meant the model was using the match being predicted to predict itself, which is obviously bad
        lambda s: s.rolling(5).mean().shift(1)) ## finds average of stat over the last 5 matches ordered by team so you can compare their last 5

teams["form_pts"] = form("Points")
teams["form_gf"] = form("GF")
teams["form_ga"] = form("GA")  ## repeats the form functions for useful stats, goals for, goals against and points

stats = ["form_pts", "form_gf", "form_ga"]
home_form = teams[teams["is_home"]].set_index("match_id")[stats].add_prefix("home_") ## only keeps the matches that were played at home
away_form = teams[~teams["is_home"]].set_index("match_id")[stats].add_prefix("away_") ## only keeps the matches that aren't at home

data = key_stats.set_index("match_id").join(home_form).join(away_form) ## data is the table that join together the initial data, and the two new tables from the perspectives of the away and home teams
data = data.dropna().reset_index(drop=True)
print(data.shape)


features = ["home_form_pts", "home_form_gf", "home_form_ga",
            "away_form_pts", "away_form_gf", "away_form_ga"]


train = data[data["Date"] < "2025-08-01"] ## I need to choose training data and data to test how accurate the model is, so I chose 2025-26 to be my test year and then the other years are my training data
test = data[data["Date"] >= "2025-08-01"]



print("Baseline:", (test["FTR"] == "H").mean()) ## we need a baseline to see how accurate our model is, the baseline I have chosen is that all matches predict Home win, as this is the most common result of the 3


model1 = LogisticRegression(max_iter=1000) ## I was looking at scikit defaults and what people normally use as their settings for this model, I found that most people use max_iter = 100, which means the model queries how wrong the data is and improves it 100 times, I wanted to increase accuracy further so I increased this by 10 times.
model1.fit(train[features], train["FTR"])
print("Logistic regression:", model1.score(test[features], test["FTR"]))


model2 = RandomForestClassifier(max_depth=5, random_state=42)
model2.fit(train[features], train["FTR"])
print("Random forest:", model2.score(test[features], test["FTR"])) ## .score takes the data and predicts results and then compares them against FTR which are the actual results giving a decimal score of what they got right

def test_prediction(home, away):
    match = test[(test["HomeTeam"] == home) & (test["AwayTeam"] == away)]
    if match.empty:
        print(f"No {home} vs {away} match found in the test data.")
        return
    real_result = match["FTR"].iloc[0] ## i tried just using [0] but that didn't work because for some reason it was looking for an element == '0', which doesn't exist so I found iloc, which actually goes to that location
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
print("Logistic Regression predictions:")
print(pd.Series(model1.predict(test[features])).value_counts())

print("Random Forest predictions:")
print(pd.Series(model2.predict(test[features])).value_counts()) ## these 4 lines tell me how many times each result is predicted by the model

test_prediction("Liverpool", "Man City")
test_prediction("Wolves","Liverpool")
test_prediction("Everton","Liverpool")
