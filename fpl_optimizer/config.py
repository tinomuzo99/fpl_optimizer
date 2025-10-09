FPL_API_BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{FPL_API_BASE}/bootstrap-static/"
FIXTURES_URL = f"{FPL_API_BASE}/fixtures/"

# Squad rules
REQUIRED_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3

# Default horizon and FDR multipliers
HORIZON_GWS = 4
DIFF_TO_MULT = {1: 1.20, 2: 1.10, 3: 1.00, 4: 0.85, 5: 0.70}

# Availability scaling
AVAIL_SCALE = {"a": 1.00, "d": 0.85, "i": 0.50, "s": 0.40, "u": 0.60}
