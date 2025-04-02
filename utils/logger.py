import os
import json
import logging
from datetime import datetime

# =========================
# LOGGING SETUP
# =========================
def setup_logger(log_dir="logs", log_file="app.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    return logging.getLogger("fake_news_logger")

logger = setup_logger()


# =========================
# CONFIGURATION MANAGEMENT
# =========================
def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def save_config(config, config_path="config.json"):
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def get_current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)


# Example Usage
if __name__ == "__main__":
    logger.info("Utility module loaded successfully.")
    
    # Example: Load configuration
    try:
        config = load_config("config.json")
        logger.info("Configuration loaded successfully.")
    except FileNotFoundError as e:
        logger.error(e)
    
    # Example: Ensure logs directory exists
    ensure_directory_exists("logs")
    logger.info("Logs directory checked/created.")