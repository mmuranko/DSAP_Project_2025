# config.py

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
    'DEFAULT': 1.50
}

# FRED Series IDs for Benchmark Rates
FRED_PROXIES = {
    'USD': 'DFF',               # Fed Funds
    'EUR': 'ECBDFR',            # ECB Deposit
    'GBP': 'IUDSOIA',           # SONIA
    'JPY': 'IRSTCI01JPM156N',   # TONAR proxy
    'CHF': 'IRSTCI01CHM156N',   # SARON
    'CAD': 'IRSTCI01CAM156N',   # CORRA proxy
    'AUD': 'RBAACTRBDAL',       # RBA Cash Rate
    'MXN': 'IRSTCI01MXM156N',   
    'NOK': 'IRSTCI01NOM156N',   
    'SEK': 'IRSTCI01SEM156N',   
    'DKK': 'IRSTCI01DKM156N',   
    'CZK': 'IRSTCI01CZM156N',   
    'HUF': 'IRSTCI01HUM156N',   
    'ILS': 'IRSTCI01ILM156N',   
    'NZD': 'IRSTCI01NZM156N',   
    'KRW': 'IRSTCI01KRM156N',   
    'ZAR': 'IRSTCI01ZAM156N',   
    'SGD': 'IRSTCI01SGM156N',   
}

# ==========================================
# 3. MARKET DATA MAPPINGS (Yahoo Finance)
# ==========================================
# Map Exchange Codes to Yahoo Ticker Suffixes
EXCHANGE_SUFFIX_MAP = {
    # Americas
    'NASDAQ': '', 'NYSE': '', 'AMEX': '', 'ARCA': '', 'PINK': '',
    'TSE': '.TO', 'VENTURE': '.V', 'MEXI': '.MX',
    # Europe
    'LSE': '.L', 'IBIS': '.DE', 'FWB': '.F', 'SBF': '.PA', 
    'AEB': '.AS', 'EBS': '.SW', 'VIRTX': '.SW', 'BM': '.MC', 
    'BVME': '.MI', 'SB': '.ST', 'SFB': '.ST', 'OSE': '.OL', 
    'CPH': '.CO', 'VIE': '.VI', 'PL': '.LS', 'EBR': '.BR',
    # Asia/Pacific
    'SEHK': '.HK', 'ASX': '.AX', 'TSEJ': '.T', 'SGX': '.SI', 
    'NSE': '.NS', 'BSE': '.BO'
}

# ==========================================
# 4. SIMULATION: FEE SCHEDULE
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
# 5. SIMULATION: DIVIDEND TAX RATES
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