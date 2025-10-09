# Fantasy Premier League Optimiser

A **Python-based optimiser** for Fantasy Premier League (FPL) squads.  
It builds an **optimal 15-player team**, recommends the **best starting XI**, and suggests **profitable single-transfer moves** — all using real-time FPL data from the public API.

---

## Quick Start (for Non-Technical Users)

1️⃣ Open **PowerShell** in your project folder (e.g. `C:\Users\<you>\Documents\fpl_optimizer`).  
2️⃣ Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

3️⃣ Install everything the project needs:
```powershell
python -m pip install -r requirements.txt
```

4️⃣ Optimise a new FPL squad (saves CSVs to `outputs/`):
```powershell
python -m fpl_optimizer.cli optimise --budget 100 --horizon 4 --allow-flagged --score-type horizon --out-dir outputs
```

5️⃣ Find your best transfer suggestions (from your own `squad.csv`):
```powershell
python -m fpl_optimizer.cli transfers --squad-file squad.csv --bank 0.5 --score-type horizon --horizon 4 --out-dir outputs
```

6️⃣ Open the `outputs` folder — you’ll find:
```
optimised_squad_horizon.csv
starting_xi_horizon.csv
transfer_recommendations_horizon.csv
```

That’s it! You now have an optimised FPL squad and smart transfer suggestions based on live data.

---

## Overview

This tool fetches live player data, fixture difficulty, and availability from the [Fantasy Premier League API](https://fantasy.premierleague.com/api/), then uses **Integer Linear Programming (ILP)** (via [PuLP](https://pypi.org/project/PuLP/)) to find the best-scoring squad under FPL rules and your specified budget.

You can:
- **Optimise a full 15-player squad** from scratch.  
- **Select the best starting XI** from your existing team.  
- **Find the best single transfer** for short- or medium-term returns.  

All outputs are saved as **CSV files** in an `outputs/` folder.

---

## 🪟 Quick Start (Developers)

### 1️⃣ Clone or download the project
```powershell
cd C:\Users\<YourUser>\Documents
git clone https://github.com/<tinomuzo99>/fpl_optimizer.git
cd fpl_optimizer
```

### 2️⃣ Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

### 3️⃣ Install dependencies
```powershell
python -m pip install -r requirements.txt
```

---

## Usage

All commands are run **from the project root** (where the `fpl_optimizer/` folder is).

You can run the CLI directly using:
```powershell
python -m fpl_optimizer.cli <command> [options]
```

---

## Commands

### 🔹 1. Optimise a new 15-player squad
```powershell
python -m fpl_optimizer.cli optimise --budget 100 --horizon 4 --allow-flagged --score-type horizon --out-dir outputs
```

#### Description
Builds the **best 15-player squad** under your budget, respecting FPL rules:
- £100m total budget  
- 15 players: 2 GK, 5 DEF, 5 MID, 3 FWD  
- ≤ 3 players per team  

#### Options

| Flag | Description |
|------|--------------|
| `--budget` | Total budget in millions (default `100`) |
| `--horizon` | Number of future gameweeks to consider (default `4`) |
| `--allow-flagged` | Include flagged/injured players (adjusted by availability multiplier) |
| `--lock` | Force-include certain players by name substring (e.g. `--lock Haaland`) |
| `--ban` | Exclude players by name substring |
| `--score-type` | Scoring metric: `next` (1 GW) or `horizon` (multi-GW weighted) |
| `--out-dir` | Directory to save CSV outputs (default `outputs/`) |

#### Example (short-term: next GW only)
```powershell
python -m fpl_optimizer.cli optimise --score-type next --budget 100 --allow-flagged --out-dir outputs
```

#### Output CSVs

| File | Description |
|------|--------------|
| `outputs/optimised_squad_horizon.csv` | Full 15-player squad with both next-GW and horizon scores |
| `outputs/starting_xi_horizon.csv` | Optimal starting XI from that squad |
| `outputs/bench_horizon.csv` | Remaining bench players |

Each player entry includes:
- `exp_points_next` — expected points for the next gameweek  
- `exp_points_h` — expected points summed over the next N (horizon) gameweeks  
- `*_avail` — versions adjusted for player availability  

---

### 🔹 2. Evaluate your current squad (best XI)
```powershell
python -m fpl_optimizer.cli xi --squad-file squad.csv --score-type horizon --horizon 4 --out-dir outputs
```

#### `squad.csv` format
You can specify just names or include a team column to disambiguate players.

**Example:**
```csv
name,team
Kelleher,Liverpool
Raya,Arsenal
Cucurella,Chelsea
Gabriel,Arsenal
Keane,Everton
Chalobah,Chelsea
Van de Ven,Tottenham
Semenyo,Bournemouth
Mbeumo,Brentford
Grealish,Manchester City
McNeil,Everton
Kevin,Manchester City
Haaland,Manchester City
Chris Wood,Nottingham Forest
Richarlison,Tottenham
```

The team column helps avoid mismatches like *Wood (Nottingham Forest)* vs *Hinshelwood (Brighton)*.

#### Output CSVs

| File | Description |
|------|--------------|
| `outputs/xi_horizon.csv` | Best starting XI |
| `outputs/bench_from_input_horizon.csv` | Remaining bench |

---

### 🔹 3. Find the best single transfer
```powershell
python -m fpl_optimizer.cli transfers --squad-file squad.csv --bank 0.5 --top-k 12 --horizon 4 --score-type horizon --out-dir outputs
```

#### Description
Finds up to `top-k` profitable single transfers for your current squad, comparing how each change affects your expected starting XI points.

#### Options

| Flag | Description |
|------|--------------|
| `--bank` | Money in the bank (default `0`) |
| `--top-k` | Show the top K transfer improvements (default `10`) |
| `--score-type` | Use `next` for short-term or `horizon` for multi-week expected points |
| `--horizon` | Horizon window (used if score-type = `horizon`) |

#### Output CSV

| File | Description |
|------|--------------|
| `outputs/transfer_recommendations_horizon.csv` | Top single-transfer upgrades and their expected point gains |

Each row includes:
- `out` / `in` — player names and teams  
- `out_pos`, `in_pos` — positions  
- `out_cost`, `in_cost` — current and replacement costs  
- `delta_pts` — expected gain in starting XI points after transfer  

---

## How scoring works

Each player has two key expected-points metrics:

| Column | Meaning |
|:--|:--|
| `exp_points_next` | Expected points for the **next gameweek** |
| `exp_points_h` | Weighted expected points over the **next N gameweeks** (set by `--horizon`) |
| `*_avail` | The same metrics adjusted for player availability (status multipliers) |

Control which to optimise with:
- `--score-type next` → short-term focus (1 GW)  
- `--score-type horizon` → medium-term planning (fixture-weighted N GWs)

Availability scaling:

| Status | Multiplier | Meaning |
|:--|--:|:--|
| `a` | 1.00 | Available |
| `d` | 0.85 | Doubtful |
| `i` | 0.50 | Injured |
| `s` | 0.40 | Suspended |
| `u` | 0.60 | Unavailable/Other |

---

## Output structure

```
outputs/
├── optimised_squad_horizon.csv
├── starting_xi_horizon.csv
├── bench_horizon.csv
├── xi_horizon.csv
├── bench_from_input_horizon.csv
└── transfer_recommendations_horizon.csv
```

---

## 🧠 Tips

- Run both **next** and **horizon** optimisations to compare short-term vs long-term picks:
  ```powershell
  python -m fpl_optimizer.cli optimise --score-type next --out-dir outputs
  python -m fpl_optimizer.cli optimise --score-type horizon --out-dir outputs
  ```
- Include `--allow-flagged` to consider players returning from injury.
- Use `--lock` to guarantee stars like Haaland stay in your squad.
- Add `--ban` to exclude players or risky teams.

---

