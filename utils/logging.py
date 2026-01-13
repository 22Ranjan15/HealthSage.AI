import logging
import os
import sys
from datetime import datetime
from config.config import Config

# 1. Create logs directory using the Global Config Path
os.makedirs(Config.LOG_DIR, exist_ok=True)

# 2. Generate unique log filename based on timestamp
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(Config.LOG_DIR, LOG_FILE_NAME)

# 3. Define Output Format
logging_str = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"

# 4. Configure Basic Config
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

# 5. Create Logger Function
def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger instance."""
    return logging.getLogger(name)
