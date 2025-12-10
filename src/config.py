# config.py

# ==========================================
# 0. CURRENCY SETTINGS
# ==========================================

RISK_FREE_RATE = 0.02

# ==========================================
# 1. CURRENCY SETTINGS
# ==========================================
# Currencies using Actual/365 day count convention
CURRENCIES_365 = [
    'AUD', 'CAD', 'GBP', 'HKD', 'KRW', 'ILS', 'INR', 'NZD', 'SGD'
]

# Currencies using Actual/360 day count convention
CURRENCIES_360 = [
    'USD', 'EUR', 'CHF', 'CZK', 'JPY', 'SEK', 'NOK', 'DKK', 'MXN', 'HUF', 'ZAR'
]

# Master list of all target currencies
TARGET_CURRENCIES = CURRENCIES_365 + CURRENCIES_360

# ==========================================
# 2. MARGIN RATE SETTINGS
# ==========================================

# Spread added to Benchmark Rate (in Percentage Points, e.g. 1.50 = 1.5%)
MARGIN_SPREADS = {
    'USD': 1.50,
    'EUR': 1.50,
    'CHF': 1.50,
    'GBP': 1.50,
    'JPY': 1.50,
    'CAD': 1.50,
    'AUD': 1.50,
    'NOK': 1.50,
    'SEK': 1.50,
    'DKK': 3.00,
    'PLN': 3.00,
    'HKD': 2.50,
    'ILS': 5.00,
    'AED': 2.50,
    'MXN': 3.00,
    'KRW': 2.00,
    'NZD': 1.50,
    'DEFAULT': 1.50
}

# FRED Series IDs for Benchmark Rates
FRED_PROXIES = {
    # Majors (Daily Data usually available)
    'USD': 'DFF',               # Fed Funds Effective Rate
    'EUR': 'ECBDFR',            # ECB Deposit Facility Rate
    'GBP': 'IUDSOIA',           # SONIA
    'JPY': 'IRSTCI01JPM156N',   # Japan Overnight Call Rate
    'CHF': 'IRSTCI01CHM156N',   # Swiss SARON
    'CAD': 'IRSTCI01CAM156N',   # Canada Overnight
    'AUD': 'IRSTCI01AUM156N',   # Immediate Rates: Call Money/Interbank Rate for Australia
    'MXN': 'IRSTCI01MXM156N',   # Mexico Overnight
    
    # Developed / OECD Nations (Monthly Averages used as Daily Proxy)
    'NOK': 'IRSTCI01NOM156N',   # Norway Interbank Rate
    'SEK': 'IRSTCI01SEM156N',   # Sweden Interbank Rate
    'DKK': 'IRSTCI01DKM156N',   # Denmark Interbank Rate
    'CZK': 'IRSTCI01CZM156N',   # Czech Rep Interbank Rate
    'HUF': 'IRSTCI01HUM156N',   # Hungary Interbank Rate
    'ILS': 'IRSTCI01ILM156N',   # Israel Interbank Rate
    'NZD': 'IRSTCI01NZM156N',   # New Zealand Interbank Rate
    'KRW': 'IRSTCI01KRM156N',   # Korea Overnight
    'ZAR': 'IRSTCI01ZAM156N',   # South Africa Interbank Rate
    
    # SGD data cannot be reliably obtained. Singapore rates track US rates closely due to MAS exchange rate management
    'SGD': 'DFF',               # Use USD Fed Funds as proxy
}

# ==========================================
# 3. MARKET DATA MAPPINGS (Yahoo Finance)
# ==========================================

# Map Exchange Codes to Yahoo Ticker Suffixes
EXCHANGE_SUFFIX_MAP = {
        # --- Americas ---
        'NASDAQ': '',     # US
        'NYSE': '',       # US
        'ARCA': '',       # US ETF
        'AMEX': '',       # US
        'PINK': '',       # US OTC
        'TSE': '.TO',     # Toronto (Canada)
        'VENTURE': '.V',  # TSX Venture (Canada)
        'MEXI': '.MX',    # Mexico

        # --- Europe ---
        'LSE': '.L',      # London (UK)
        'IBIS': '.DE',    # Xetra (Germany) - Note: IBKR uses IBIS for Xetra
        'FWB': '.F',      # Frankfurt (Germany)
        'SBF': '.PA',     # Euronext Paris (France)
        'AEB': '.AS',     # Euronext Amsterdam (Netherlands)
        'EBS': '.SW',     # SIX Swiss Exchange (Switzerland)
        'VIRTX': '.SW',   # SIX Swiss (Blue chips)
        'BM': '.MC',      # Bolsa de Madrid (Spain)
        'BVME': '.MI',    # Borsa Italiana (Italy)
        'SB': '.ST',      # Stockholm (Sweden)
        'SFB': '.ST',     # Stockholm (Sweden)
        'OSE': '.OL',     # Oslo (Norway)
        'CPH': '.CO',     # Copenhagen (Denmark)
        'VIE': '.VI',     # Vienna (Austria)
        'PL': '.LS',      # Lisbon (Portugal)
        'EBR': '.BR',     # Brussels (Belgium)

        # --- Asia / Pacific ---
        'SEHK': '.HK',    # Hong Kong
        'ASX': '.AX',     # Australia
        'TSEJ': '.T',     # Tokyo (Japan)
        'SGX': '.SI',     # Singapore
        'NSE': '.NS',     # India (National)
        'BSE': '.BO',     # India (Bombay)
    }

# ==========================================
# 4. FEE SCHEDULE
# ==========================================

# 'fee': Fixed cost in Asset Currency
# 'tax_buy': Percentage tax on PURCHASE only
FEE_SCHEDULE = {
    # US
    'NASDAQ': {'fee': 1.0, 'tax_buy': 0.0},
    'NYSE':   {'fee': 1.0, 'tax_buy': 0.0},
    'AMEX':   {'fee': 1.0, 'tax_buy': 0.0},
    'ARCA':   {'fee': 1.0, 'tax_buy': 0.0},
    'PINK':   {'fee': 1.0, 'tax_buy': 0.0},
    # CH
    'EBS':    {'fee': 5.0, 'tax_buy': 0.0},
    'VIRTX':  {'fee': 5.0, 'tax_buy': 0.0},
    # Europe
    'SBF':    {'fee': 3.0, 'tax_buy': 0.004}, # FR
    'LSE':    {'fee': 3.0, 'tax_buy': 0.005}, # UK
    'IBIS':   {'fee': 3.0, 'tax_buy': 0.0},   # DE
    'FWB':    {'fee': 3.0, 'tax_buy': 0.0},   # DE
    'AEB':    {'fee': 3.0, 'tax_buy': 0.0},   # NL
    'BM':     {'fee': 3.0, 'tax_buy': 0.0},   # ES
    'BVME':   {'fee': 3.0, 'tax_buy': 0.0},   # IT
    'PL':     {'fee': 3.0, 'tax_buy': 0.0},   # PT
    'EBR':    {'fee': 3.0, 'tax_buy': 0.0},   # BE
    # Default
    'DEFAULT': {'fee': 4.0, 'tax_buy': 0.0} 
}

# ==========================================
# 5. DIVIDEND TAX RATES
# ==========================================

# Withholding Tax Rates by Exchange
DIV_TAX_RATES = {
    'NASDAQ': 0.15, 'NYSE': 0.15, 'AMEX': 0.15, 'ARCA': 0.15, 'PINK': 0.15,
    'EBS': 0.35, 'VIRTX': 0.35,
    'SBF': 0.25,
    'IBIS': 0.26375, 'FWB': 0.26375,
    'AEB': 0.17,
    'BM': 0.19,
    'BVME': 0.26,
    'LSE': 0.00,
    'DEFAULT': 0.15
}