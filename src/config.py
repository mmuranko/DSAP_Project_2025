# ==========================================
# 0. RISK-FREE RATE SETTING
# ==========================================

RISK_FREE_RATE = 0.0034 # Swiss Confederation 10-Year Government Bond, Dec 2025

# ==========================================
# 1. CURRENCY SETTINGS
# ==========================================

# Currencies using Actual/365 day count convention
CURRENCIES_365 = [
    'AUD', 
    'CAD', 
    'GBP', 
    'HKD', 
    'KRW', 
    'ILS', 
    'INR', 
    'NZD', 
    'SGD'
]

# Currencies using Actual/360 day count convention
CURRENCIES_360 = [
    'USD', 
    'EUR', 
    'CHF', 
    'CZK', 
    'JPY', 
    'SEK', 
    'NOK', 
    'DKK', 
    'MXN', 
    'HUF', 
    'ZAR'
]

# Master list of all target currencies
TARGET_CURRENCIES = CURRENCIES_365 + CURRENCIES_360

# ==========================================
# 2. MARGIN RATE SETTINGS
# ==========================================

# Spread added to Benchmark Rate (in Percentage Points, e.g. 1.50 = 1.5%)
MARGIN_SPREADS = {
    'DEFAULT': 1.50,
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
    'NZD': 1.50
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
        'MEXI': '.MX',    # Mexico

        # --- Europe ---
        'EBS': '.SW',     # SIX Swiss Exchange (Switzerland)
        'VIRTX': '.SW',   # SIX Swiss (Blue chips)
        'LSE': '.L',      # London (UK)
        'IBIS': '.DE',    # Xetra (Germany)
        'FWB': '.F',      # Frankfurt (Germany)
        'SBF': '.PA',     # Euronext Paris (France)
        'AEB': '.AS',     # Euronext Amsterdam (Netherlands)
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
        'BSE': '.BO'     # India (Bombay)
    }

# ==========================================
# 4. FEE SCHEDULE
# ==========================================

# 'fee': Fixed cost in Asset Currency
# 'tax_buy': Percentage tax on PURCHASE only
FEE_SCHEDULE = {
    # --- Default ---
    'DEFAULT': {'fee': 5.0, 'tax_buy': 0.0},

    # --- Americas ---
    'NASDAQ': {'fee': 1.0, 'tax_buy': 0.0},     # US
    'NYSE': {'fee': 1.0, 'tax_buy': 0.0},       # US
    'ARCA': {'fee': 1.0, 'tax_buy': 0.0},       # US ETF
    'AMEX': {'fee': 1.0, 'tax_buy': 0.0},       # US
    'PINK': {'fee': 1.0, 'tax_buy': 0.0},       # US OTC
    'TSE': {'fee': 1.0, 'tax_buy': 0.0},     # Canada
    'MEXI': {'fee': 60.0, 'tax_buy': 0.0},    # Mexico
    
    # --- Europe ---
    'EBS': {'fee': 5.0, 'tax_buy': 0.0}, # CH
    'VIRTX': {'fee': 5.0, 'tax_buy': 0.0}, # CH
    'LSE': {'fee': 3.0, 'tax_buy': 0.005}, # UK
    'IBIS': {'fee': 3.0, 'tax_buy': 0.0},   # DE
    'FWB': {'fee': 3.0, 'tax_buy': 0.0},   # DE
    'SBF': {'fee': 3.0, 'tax_buy': 0.004}, # FR
    'AEB': {'fee': 3.0, 'tax_buy': 0.0},   # NL
    'BM': {'fee': 3.0, 'tax_buy': 0.0},   # ES
    'BVME': {'fee': 3.0, 'tax_buy': 0.001},   # IT
    'SB': {'fee': 49.0, 'tax_buy': 0.0},   # SW
    'SFB': {'fee': 49.0, 'tax_buy': 0.0},   # SW
    'OSE': {'fee': 49.0, 'tax_buy': 0.0},   # NO
    'CPH': {'fee': 49.0, 'tax_buy': 0.0},   # DK
    'VIE': {'fee': 3.0, 'tax_buy': 0.0},   # AT
    'PL': {'fee': 3.0, 'tax_buy': 0.0},   # PT
    'EBR': {'fee': 3.0, 'tax_buy': 0.0012},   # BE

    # --- Asia / Pacific ---
    'SEHK': {'fee': 18.0, 'tax_buy': 0.001}, # Hong-Kong
    'ASX': {'fee': 6.0, 'tax_buy': 0.0}, # Australia
    'TSEJ': {'fee': 80.0, 'tax_buy': 0.0}, # Japan
    'SGX': {'fee': 2.5, 'tax_buy': 0.0}, # Singapore
    'NSE': {'fee': 20.0, 'tax_buy': 0.0}, # India (National)
    'BSE': {'fee': 20.0, 'tax_buy': 0.0} # India (Bombay)
}

# ==========================================
# 5. DIVIDEND TAX RATES
# ==========================================

# Withholding Tax Rates by Exchange
DIVIDEND_TAX_RATES = {
    # --- Default ---
    'DEFAULT': 0.35,

    # --- Americas ---
    'NASDAQ': 0.15,      # US (User specified 15%, Statutory is 30%)
    'NYSE': 0.15,        # US
    'ARCA': 0.15,        # US
    'AMEX': 0.15,        # US
    'PINK': 0.15,        # US
    'TSE': 0.25,         # Canada (Statutory 25%, often 15% w/ treaty)
    'MEXI': 0.10,        # Mexico (10%)

    # --- Europe ---
    'EBS': 0.35,         # Switzerland (35% - very high strict WHT)
    'VIRTX': 0.35,       # Switzerland
    'LSE': 0.00,         # UK (0% standard for non-residents)
    'IBIS': 0.26375,     # Germany (25% + 5.5% Solidarity Surcharge)
    'FWB': 0.26375,      # Germany
    'SBF': 0.25,         # France (25% standard non-resident corporate rate)
    'AEB': 0.15,         # Netherlands (15%)
    'BM': 0.19,          # Spain (19%)
    'BVME': 0.26,        # Italy (26%)
    'SB': 0.30,          # Sweden (30%)
    'SFB': 0.30,         # Sweden
    'OSE': 0.25,         # Norway (25%)
    'CPH': 0.27,         # Denmark (27%)
    'VIE': 0.275,        # Austria (27.5%)
    'PL': 0.28,          # Portugal (28%)
    'EBR': 0.30,         # Belgium (30%)

    # --- Asia / Pacific ---
    'SEHK': 0.00,        # Hong Kong (0%)
    'ASX': 0.30,         # Australia (30%)
    'TSEJ': 0.15,        # Japan (15%)
    'SGX': 0.00,         # Singapore (0%)
    'NSE': 0.20,         # India (20%)
    'BSE': 0.20          # India
}