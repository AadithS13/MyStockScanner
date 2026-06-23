"""Indian defence-sector universe for the AI prediction prototype.

Each entry: NSE symbol -> (display name, Yahoo Finance ticker).
We start focused: liquid, established names with enough price history
to train on. Recent IPOs with <1yr of data are intentionally excluded
for now (we can add them once they season).
"""

DEFENCE_STOCKS = {
    "HAL":        ("Hindustan Aeronautics",      "HAL.NS"),
    "BEL":        ("Bharat Electronics",         "BEL.NS"),
    "BDL":        ("Bharat Dynamics",            "BDL.NS"),
    "MAZDOCK":    ("Mazagon Dock Shipbuilders",  "MAZDOCK.NS"),
    "COCHINSHIP": ("Cochin Shipyard",            "COCHINSHIP.NS"),
    "GRSE":       ("Garden Reach Shipbuilders",  "GRSE.NS"),
    "DATAPATTNS": ("Data Patterns",              "DATAPATTNS.NS"),
    "PARAS":      ("Paras Defence & Space",      "PARAS.NS"),
    "ZENTEC":     ("Zen Technologies",           "ZENTEC.NS"),
    "SOLARINDS":  ("Solar Industries",           "SOLARINDS.NS"),
    "MTARTECH":   ("MTAR Technologies",          "MTARTECH.NS"),
    "ASTRAMICRO": ("Astra Microwave Products",   "ASTRAMICRO.NS"),
    "BEML":       ("BEML Ltd",                   "BEML.NS"),
    "DYNAMATECH": ("Dynamatic Technologies",     "DYNAMATECH.NS"),
    "HBLENGINE":  ("HBL Engineering",            "HBLENGINE.NS"),
    "IDEAFORGE":  ("ideaForge Technology",       "IDEAFORGE.NS"),
}

# Symbol -> Yahoo ticker
YF_TICKERS = {sym: yf for sym, (_, yf) in DEFENCE_STOCKS.items()}
# Symbol -> display name
NAMES = {sym: name for sym, (name, _) in DEFENCE_STOCKS.items()}


def symbols() -> list[str]:
    return list(DEFENCE_STOCKS.keys())
