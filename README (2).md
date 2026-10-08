Premier League Match Prediction

Can recent form predict the result of a Premier League match (home win, draw or away win)? This project trains two models on four seasons of results and tests them on a fifth.

Findings

I tested on the 2025-26 season (375 matches). The baseline always predicts a home win, which was by far the most common result.

Model Accuracy:

Baseline (always home win): 42.4% 
Logistic regression: 46.1% 
Random forest: 44.5% 

Number of each result the models predicted:

| Result | Logistic regression | Random forest |
|---|---|---|
| Home win | 253 | 281 |
| Away win | 122 | 93 |
| Draw | 0 | 1 |

- Both models beat the baseline, but only slightly. With 375 test matches, gaps of a few percentage points could easily be noise, including the gap between the two models.
- The models almost never predict a draw, even though draws make up roughly a quarter of Premier League results. A draw is rarely the single most likely outcome of a match, so a model that picks the most likely result will nearly always choose a home or away win.
- Overall, recent results and goals only help a little. To do better I would need more information than just results, home goals and away goals.

How it works

- Data: Premier League results from 2021-22 to 2025-26, from [Football-Data.co.uk](https://www.football-data.co.uk).
- Features: each team's average points, goals scored and goals conceded over its previous 5 matches, for both the home and away side (6 features). The averages only use earlier matches, so the model never sees the result it is predicting.
- Split: trained on 2021-22 to 2024-25, tested on 2025-26. The split is by date, not random, to mimic predicting future matches.
- Models: logistic regression, and a random forest (many decision trees, each asking up to 5 questions in a row, with the most common answer winning).
- Baseline: always predict a home win.

Limitations and next steps

- Only six features, and they ignore the strength of the opponent.
- Accuracy hides how badly draws are predicted. Next I would add a per-class report and try class weighting.
- Compare against bookmaker odds and add features such as Elo ratings.
