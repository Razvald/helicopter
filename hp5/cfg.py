# cfg.py

# Mavlink port configuration
PORT = 'udp:127.0.0.1:14550'

# Dump directory and settings
DUMP_DIR = '/home/orangepi/dumps'
MIN_FREE_SPACE = 2**31  # bytes

# Flags for dumping and sending data
DUMP = True

# GPS mode configuration
GPS_MODE = 'vio'  # Options: 'gps', 'vio'

# Default GPS coordinates (used if GPS data is not available)
DEFAULT_LAT = 54.84309569281793
DEFAULT_LON = 83.09851770880381
DEFAULT_ALT = 204
