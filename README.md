# Fantasy Premier League Optimiser

A Python optimiser for Fantasy Premier League (FPL) squad selection, starting-XI selection, captaincy, transfer analysis, historical backtesting, and weekly performance tracking.

The project uses live data from the public FPL API and Integer Linear Programming through PuLP to produce legal FPL squads under configurable budget and selection constraints.

## Features

The optimiser can:

- Select a legal 15-player FPL squad.
- Select a legal starting XI.
- Choose a captain and vice-captain.
- Compare sequential and joint squad-selection strategies.
- Use next-gameweek, multi-gameweek, or validated scoring.
- Recommend beneficial single-player transfers.
- Backtest scoring methods over historical seasons.
- Save weekly recommendations before the deadline.
- Fetch actual gameweek points after the gameweek finishes.
- Apply captain fallback and legal automatic substitutions.
- Compare predicted and actual weekly points.

## Installation

Open PowerShell in the project directory.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the environment

```powershell
.\.venv\Scripts\Activate
```

### 3. Install the dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the tests

```powershell
python -m pytest -q
```

## Command-line interface

All commands should be run from the project root:

```powershell
python -m fpl_optimizer.cli <command> [options]
```

The available commands are:

| Command | Purpose |
|---|---|
| `optimise` | Generate a new 15-player squad, starting XI, captain and vice-captain |
| `xi` | Select the best XI from an existing 15-player squad |
| `transfers` | Recommend beneficial single-player transfers |
| `track` | Score saved recommendations after a gameweek finishes |

## Generate a recommended squad

The recommended live workflow uses:

- The validated scoring method
- The sequential squad-selection strategy
- Five recent gameweeks for expected minutes
- Five full-match equivalents for positional shrinkage

```powershell
python -m fpl_optimizer.cli optimise `
  --selection-strategy sequential `
  --score-type validated `
  --recent-gws 5 `
  --shrinkage-matches 5 `
  --budget 100 `
  --allow-flagged `
  --out-dir outputs-validated
```

The optimiser enforces the standard squad constraints:

- 15 players
- 2 goalkeepers
- 5 defenders
- 5 midfielders
- 3 forwards
- No more than 3 players from one club
- Total cost within the specified budget

It also selects a legal starting XI containing:

- Exactly 1 goalkeeper
- At least 3 defenders
- At least 2 midfielders
- At least 1 forward
- Exactly 11 players in total

### Selection strategies

The `--selection-strategy` option supports two approaches.

#### Sequential strategy

```powershell
--selection-strategy sequential
```

This is the default and currently recommended strategy.

It:

1. Selects the best legal 15-player squad.
2. Selects the best legal XI from that squad.
3. Selects the captain and vice-captain from the starting XI.

#### Joint strategy

```powershell
--selection-strategy joint --bench-weight 0.10
```

This experimental strategy jointly selects the squad, starting XI and captain.

The `--bench-weight` argument controls how much value is assigned to substitute points relative to starting-XI points.

## Scoring methods

The `--score-type` option supports three methods:

| Value | Description |
|---|---|
| `next` | Availability-adjusted expected points for the next gameweek |
| `horizon` | Fixture-weighted expected points over multiple future gameweeks |
| `validated` | Minutes-and-shrinkage model validated through historical backtesting |

### Next-gameweek scoring

```powershell
python -m fpl_optimizer.cli optimise `
  --score-type next `
  --budget 100 `
  --out-dir outputs-next
```

The optimiser uses:

```text
exp_points_next_avail
```

### Horizon scoring

```powershell
python -m fpl_optimizer.cli optimise `
  --score-type horizon `
  --horizon 4 `
  --budget 100 `
  --out-dir outputs-horizon
```

The optimiser uses:

```text
exp_points_h_avail
```

### Validated live scoring

```powershell
python -m fpl_optimizer.cli optimise `
  --score-type validated `
  --recent-gws 5 `
  --shrinkage-matches 5 `
  --budget 100 `
  --out-dir outputs-validated
```

The validated model estimates a player's next-gameweek score using:

- Recent minutes
- Recent team fixture count
- Expected minutes share
- Season points per 90
- A position-level points-per-90 prior
- Shrinkage toward the position prior
- Next-gameweek fixture difficulty
- Player availability

The final score is stored in:

```text
exp_points_validated_live
```

The diagnostic columns include:

| Column | Meaning |
|---|---|
| `recent_minutes` | Minutes played during the recent gameweek window |
| `recent_fixture_count` | Number of fixtures played by the player's club during that window |
| `expected_minutes_share` | Estimated proportion of available match minutes |
| `position_points_per_90` | Average points per 90 for the player's position |
| `shrunk_points_per_90` | Player rate after positional shrinkage |
| `next_fixture_weight` | Combined weight of the next gameweek's fixtures |
| `exp_points_validated_live` | Final validated live projection |
| `validated_score_source` | Indicates whether the model or a fallback score was used |

Before three gameweeks have been completed, there is not enough current-season information for the minutes model. During this period, the optimiser uses the official next-gameweek estimate.

Players without sufficient current-season history also receive an official-score fallback.

## Availability adjustment

Player projections are adjusted using availability multipliers.

| Status | Multiplier | Meaning |
|---|---:|---|
| `a` | 1.00 | Available |
| `d` | 0.85 | Doubtful |
| `i` | 0.50 | Injured |
| `s` | 0.40 | Suspended |
| `u` | 0.60 | Unavailable or other |

Use `--allow-flagged` to keep flagged players in the candidate pool while applying the relevant availability penalty.

## Optimisation outputs

A validated optimisation creates:

```text
outputs-validated/
  optimised_squad_validated.csv
  starting_xi_validated.csv
  bench_validated.csv
```

The squad file records the assigned role for every player:

- `CAPTAIN`
- `VICE`
- `XI`
- `BENCH`

The console summary reports:

- Total squad cost
- Captain
- Vice-captain
- Expected starting-XI points
- Captain bonus
- Projected FPL points
- Joint objective value when the joint strategy is used

Projected FPL points are calculated as:

```text
starting-XI expected points + captain expected points
```

The captain's expected points are added once more to represent captain doubling.

## Weekly recommendation tracking

The optimiser can preserve one or more recommendations for each gameweek and score them after the gameweek finishes.

### Record a recommendation

Add `--track` when generating the team you want to evaluate:

```powershell
python -m fpl_optimizer.cli optimise `
  --selection-strategy sequential `
  --score-type validated `
  --recent-gws 5 `
  --shrinkage-matches 5 `
  --budget 100 `
  --allow-flagged `
  --track `
  --tracking-dir tracking `
  --out-dir outputs-validated
```

Only commands containing `--track` are added to the permanent weekly record. This allows other optimisation runs to remain disposable experiments.

Each tracked recommendation receives a unique `run_id`.

### Score a finished gameweek

After FPL marks the gameweek as finished, run:

```powershell
python -m fpl_optimizer.cli track `
  --tracking-dir tracking
```

The tracker fetches official player minutes and points from the FPL event-live API.

It then:

1. Identifies players who recorded zero minutes.
2. Applies the substitute goalkeeper when required.
3. Processes outfield substitutes in bench order.
4. Ensures every automatic substitution preserves a legal formation.
5. Transfers captaincy to the vice-captain when the captain records zero minutes.
6. Calculates the final recommended-team score.

### Tracking outputs

```text
tracking/
  weekly_summary.csv
  weekly_picks.csv
```

#### `weekly_summary.csv`

Contains one row per tracked recommendation, including:

- Season
- Target gameweek
- Selection strategy
- Scoring method
- Model parameters
- Predicted XI points
- Predicted captain bonus
- Predicted FPL points
- Actual XI points
- Actual bench points
- Actual captain bonus
- Actual FPL points
- Prediction error
- Captain who counted
- Number of automatic substitutions
- Tracking status

Prediction error is calculated as:

```text
actual FPL points - predicted FPL points
```

#### `weekly_picks.csv`

Contains all 15 players for every tracked recommendation, including:

- Player and club
- Position
- Assigned role
- Bench order
- Cost
- Predicted points
- Actual minutes
- Actual points
- Whether the player counted in the final XI

The tracker measures the raw score of the saved recommendation. It does not include:

- Transfer-point deductions
- Chips
- Later manual squad changes
- Manual captaincy changes made outside the optimiser

## Evaluate an existing squad

Create a CSV containing 15 players.

The required column is:

```text
name
```

An optional `team` or `team_name` column can be included to disambiguate players with similar names.

Example structure:

```csv
name,team
Player Name,Club Name
```

Run:

```powershell
python -m fpl_optimizer.cli xi `
  --squad-file squad.csv `
  --score-type validated `
  --recent-gws 5 `
  --shrinkage-matches 5 `
  --out-dir outputs-xi
```

The command:

- Validates the 15-player squad.
- Selects a legal starting XI.
- Selects a captain and vice-captain.
- Saves the XI and bench to CSV files.

Outputs:

```text
outputs-xi/
  xi_validated.csv
  bench_from_input_validated.csv
```

## Transfer recommendations

To evaluate single-player transfers:

```powershell
python -m fpl_optimizer.cli transfers `
  --squad-file squad.csv `
  --bank 0.5 `
  --top-k 10 `
  --score-type validated `
  --recent-gws 5 `
  --shrinkage-matches 5 `
  --out-dir outputs-transfers
```

Each proposed transfer:

- Replaces a player with another player in the same position.
- Respects the available bank balance.
- Preserves the three-player-per-club limit.
- Produces a legal 15-player squad.
- Recalculates the starting XI.
- Recalculates captaincy.
- Reports the expected improvement in projected FPL points.

## Historical backtesting

The backtesting pipeline evaluates model performance using archived FPL seasons.

Historical data is downloaded from the public `vaastav/Fantasy-Premier-League` dataset.

### Download a historical season

```powershell
python scripts/download_historical_data.py `
  --season 2024-25 `
  --out-dir data/historical
```

This downloads:

```text
data/historical/2024-25/
  merged_gw.csv
  fixtures.csv
  teams.csv
```

### Run a backtest

```powershell
python -m scripts.run_backtest `
  --season 2024-25 `
  --start-gw 4 `
  --end-gw 38 `
  --budget 100 `
  --recent-gws 5 `
  --shrinkage-matches 5 `
  --selection-strategy sequential `
  --out-dir backtests/results/2024-25
```

The default validated-model parameters are:

```text
recent gameweeks: 5
shrinkage matches: 5
```

These parameters were kept unchanged when evaluating the 2022-23, 2023-24, and 2024-25 seasons.

### Backtested models

| Model | Description |
|---|---|
| `prior_ppg` | Points per appearance before the target gameweek |
| `fixture_ppg` | Prior points per appearance adjusted for fixture difficulty |
| `minutes_shrunk_fixture` | Position-shrunk points per 90 combined with recent expected minutes and fixture difficulty |

The backtest uses only information available before each target gameweek when constructing player performance features.

The archived expected-points field is deliberately excluded because it may have been captured after the matches and could introduce future-information leakage.

### Backtest outputs

```text
backtests/results/2024-25/
  player_predictions.csv
  gameweek_results.csv
  summary.csv
  paired_comparisons.csv
```

| File | Contents |
|---|---|
| `player_predictions.csv` | Player-level predictions, actual scores, and squad/XI selections |
| `gameweek_results.csv` | Model metrics and selected-team performance for each gameweek |
| `summary.csv` | Average performance metrics for each model |
| `paired_comparisons.csv` | Paired XI-point comparisons with bootstrap confidence intervals |

The reported metrics include:

- Mean absolute error
- Root mean squared error
- Spearman rank correlation
- Top-K recall
- Actual starting-XI points
- Actual squad points
- Actual bench points
- Predicted starting-XI points
- Paired model wins and losses
- 95% bootstrap confidence intervals

The historical backtest does not currently simulate:

- Transfer histories
- Selling-price changes
- Transfer-point deductions
- Chips
- Historical injury flags
- Full automatic-substitution behaviour

Archived fixture difficulty may also differ from the value that was available at the original gameweek deadline.

## Testing

Run the complete test suite with:

```powershell
python -m pytest -q
```

The tests cover:

- API snapshot compatibility
- Legal squad construction
- Legal starting-XI selection
- Captain and vice-captain selection
- Joint optimisation
- Historical backtesting
- Squad validation
- Player-name matching

PuLP may display deprecation warnings relating to its upcoming 4.0 API. These warnings do not currently indicate test failures.

## Project structure

```text
fpl_optimizer/
  backtest.py
  cli.py
  config.py
  fpl.py
  name_matching.py
  optimizer.py
  tracking.py

scripts/
  download_historical_data.py
  run_backtest.py
  update_api_snapshots.py

tests/
  snapshots/
  test_api_snapshot.py
  test_backtest.py
  test_validation.py

README.md
requirements.txt
squad.csv
```

Generated outputs, tracking records, downloaded historical data, and backtest results are excluded from version control.

## Recommended weekly workflow

Before the gameweek deadline:

```powershell
python -m fpl_optimizer.cli optimise `
  --selection-strategy sequential `
  --score-type validated `
  --recent-gws 5 `
  --shrinkage-matches 5 `
  --budget 100 `
  --allow-flagged `
  --track `
  --tracking-dir tracking `
  --out-dir outputs-validated
```

After the gameweek is marked finished:

```powershell
python -m fpl_optimizer.cli track `
  --tracking-dir tracking
```

Review the weekly results:

```powershell
Import-Csv tracking/weekly_summary.csv |
  Format-Table `
    season, `
    target_gw, `
    predicted_fpl_points, `
    actual_fpl_points, `
    prediction_error, `
    captain_counted, `
    autosub_count
```

## Data sources

- Live data: [Fantasy Premier League API](https://fantasy.premierleague.com/api/)
- Historical data: [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
- Optimisation library: [PuLP](https://pypi.org/project/PuLP/)

## Licence

Add the appropriate project licence here if the repository will be distributed or reused publicly.