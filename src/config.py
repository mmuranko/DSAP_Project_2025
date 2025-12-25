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

# Spread added to Benchmark Rate
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
    'USD': 'DFF',                   # Fed Funds Effective Rate
    'EUR': 'ECBDFR',                # ECB Deposit Facility Rate
    'GBP': 'IUDSOIA',               # SONIA
    'JPY': 'IRSTCI01JPM156N',       # Japan Overnight Call Rate
    'CHF': 'IRSTCI01CHM156N',       # Swiss SARON
    'CAD': 'IRSTCI01CAM156N',       # Canada Overnight
    'AUD': 'IRSTCI01AUM156N',       # Immediate Rates: Call Money/Interbank Rate for Australia
    'MXN': 'IRSTCI01MXM156N',       # Mexico Overnight
    
    # Developed / OECD Nations (Monthly Averages used as Daily Proxy)
    'NOK': 'IRSTCI01NOM156N',       # Norway Interbank Rate
    'SEK': 'IRSTCI01SEM156N',       # Sweden Interbank Rate
    'DKK': 'IRSTCI01DKM156N',       # Denmark Interbank Rate
    'CZK': 'IRSTCI01CZM156N',       # Czech Rep Interbank Rate
    'HUF': 'IRSTCI01HUM156N',       # Hungary Interbank Rate
    'ILS': 'IRSTCI01ILM156N',       # Israel Interbank Rate
    'NZD': 'IRSTCI01NZM156N',       # New Zealand Interbank Rate
    'KRW': 'IRSTCI01KRM156N',       # Korea Overnight
    'ZAR': 'IRSTCI01ZAM156N',       # South Africa Interbank Rate
    'SGD': 'DFF',                   # Hard to get, use USD Fed Funds as proxy
}

# ==========================================
# 3. MARKET DATA MAPPINGS (Yahoo Finance)
# ==========================================
EXCHANGE_SUFFIX_MAP = {
        # --- Americas ---
        'NASDAQ': '',       # USA
        'NYSE': '',         # USA
        'ARCA': '',         # USA
        'AMEX': '',         # USA
        'PINK': '',         # USA
        'TSE': '.TO',       # Canada
        'MEXI': '.MX',      # Mexico

        # --- Europe ---
        'EBS': '.SW',       # Switzerland
        'VIRTX': '.SW',     # Switzerland
        'LSE': '.L',        # UK
        'IBIS': '.DE',      # Germany
        'FWB': '.F',        # Germany
        'SBF': '.PA',       # France
        'AEB': '.AS',       # Netherlands
        'BM': '.MC',        # Spain
        'BVME': '.MI',      # Italy
        'SB': '.ST',        # Sweden
        'SFB': '.ST',       # Sweden
        'OSE': '.OL',       # Norway
        'CPH': '.CO',       # Denmark
        'VIE': '.VI',       # Austria
        'PL': '.LS',        # Portugal
        'EBR': '.BR',       # Belgium

        # --- Asia / Pacific ---
        'SEHK': '.HK',      # Hong Kong
        'ASX': '.AX',       # Australia
        'TSEJ': '.T',       # Japan
        'SGX': '.SI',       # Singapore
        'NSE': '.NS',       # India 
        'BSE': '.BO'        # India
    }

# ==========================================
# 4. FEE SCHEDULE
# ==========================================
FEE_SCHEDULE = {
    # --- Default ---
    'DEFAULT': {'fee': 5.0, 'tax_buy': 0.0},

    # --- Americas ---
    'NASDAQ': {'fee': 1.0, 'tax_buy': 0.0},     # USA
    'NYSE': {'fee': 1.0, 'tax_buy': 0.0},       # USA
    'ARCA': {'fee': 1.0, 'tax_buy': 0.0},       # USA
    'AMEX': {'fee': 1.0, 'tax_buy': 0.0},       # USA
    'PINK': {'fee': 1.0, 'tax_buy': 0.0},       # USA
    'TSE': {'fee': 1.0, 'tax_buy': 0.0},        # Canada
    'MEXI': {'fee': 60.0, 'tax_buy': 0.0},      # Mexico
    
    # --- Europe ---
    'EBS': {'fee': 5.0, 'tax_buy': 0.0},        # Switzerland
    'VIRTX': {'fee': 5.0, 'tax_buy': 0.0},      # Switzerland
    'LSE': {'fee': 3.0, 'tax_buy': 0.005},      # UK
    'IBIS': {'fee': 3.0, 'tax_buy': 0.0},       # Germany
    'FWB': {'fee': 3.0, 'tax_buy': 0.0},        # Germany
    'SBF': {'fee': 3.0, 'tax_buy': 0.004},      # France
    'AEB': {'fee': 3.0, 'tax_buy': 0.0},        # Netherlands
    'BM': {'fee': 3.0, 'tax_buy': 0.0},         # Spain
    'BVME': {'fee': 3.0, 'tax_buy': 0.001},     # Italy
    'SB': {'fee': 49.0, 'tax_buy': 0.0},        # Sweden
    'SFB': {'fee': 49.0, 'tax_buy': 0.0},       # Sweden
    'OSE': {'fee': 49.0, 'tax_buy': 0.0},       # Norway
    'CPH': {'fee': 49.0, 'tax_buy': 0.0},       # Denmark
    'VIE': {'fee': 3.0, 'tax_buy': 0.0},        # Austria
    'PL': {'fee': 3.0, 'tax_buy': 0.0},         # Portugal
    'EBR': {'fee': 3.0, 'tax_buy': 0.0012},     # Belgium

    # --- Asia / Pacific ---
    'SEHK': {'fee': 18.0, 'tax_buy': 0.001},    # Hong-Kong
    'ASX': {'fee': 6.0, 'tax_buy': 0.0},        # Australia
    'TSEJ': {'fee': 80.0, 'tax_buy': 0.0},      # Japan
    'SGX': {'fee': 2.5, 'tax_buy': 0.0},        # Singapore
    'NSE': {'fee': 20.0, 'tax_buy': 0.0},       # India
    'BSE': {'fee': 20.0, 'tax_buy': 0.0}        # India
}

# ==========================================
# 5. DIVIDEND TAX RATES
# ==========================================
DIVIDEND_TAX_RATES = {
    # --- Default ---
    'DEFAULT': 0.35,

    # --- Americas ---
    'NASDAQ': 0.15,      # USA
    'NYSE': 0.15,        # USA
    'ARCA': 0.15,        # USA
    'AMEX': 0.15,        # USA
    'PINK': 0.15,        # USA
    'TSE': 0.25,         # Canada
    'MEXI': 0.10,        # Mexico

    # --- Europe ---
    'EBS': 0.35,         # Switzerland
    'VIRTX': 0.35,       # Switzerland
    'LSE': 0.00,         # UK
    'IBIS': 0.26375,     # Germany
    'FWB': 0.26375,      # Germany
    'SBF': 0.25,         # France
    'AEB': 0.15,         # Netherlands
    'BM': 0.19,          # Spain
    'BVME': 0.26,        # Italy
    'SB': 0.30,          # Sweden
    'SFB': 0.30,         # Sweden
    'OSE': 0.25,         # Norway
    'CPH': 0.27,         # Denmark
    'VIE': 0.275,        # Austria
    'PL': 0.28,          # Portugal
    'EBR': 0.30,         # Belgium

    # --- Asia / Pacific ---
    'SEHK': 0.00,        # Hong Kong
    'ASX': 0.30,         # Australia
    'TSEJ': 0.15,        # Japan
    'SGX': 0.00,         # Singapore
    'NSE': 0.20,         # India
    'BSE': 0.20          # India
}