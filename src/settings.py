from datetime import datetime
from pathlib import Path
import logging


# Project structure
HDD = Path('/media/gunes/d0c7ecce-dc82-4c90-a953-1dc746490bf5/')
ROOT = Path('/home/gunes/Desktop/Kaggle/ubc-ovarian-cancer-subtype-classification-and-outlier-detection')
DATA = ROOT / 'data'
LOGS = ROOT / 'logs'
MODELS = ROOT / 'models'
EDA = ROOT / 'eda'

# Logging configurations
LOGGING_LEVEL = logging.INFO
log_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log_file_handler = logging.FileHandler(filename=LOGS / f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log')
log_file_handler.setFormatter(log_formatter)
log_file_handler.setLevel(LOGGING_LEVEL)
log_stream_handler = logging.StreamHandler()
log_stream_handler.setFormatter(log_formatter)
log_stream_handler.setLevel(LOGGING_LEVEL)
logger = logging.getLogger('root')
logger.setLevel(LOGGING_LEVEL)
logger.addHandler(log_file_handler)
logger.addHandler(log_stream_handler)
