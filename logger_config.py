import logging

# Configure the logging
logging.basicConfig(
    filename='qwen50-100.log',        # Shared log file
    filemode='w',              # mode
    level=logging.DEBUG,       # Set log level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Optionally create a named logger
logger = logging.getLogger("shared_logger")

