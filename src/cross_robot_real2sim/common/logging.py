import logging
import time

def configure_logging():
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(name)s %(message)s", force=True)
    return logging.getLogger("cross_robot_real2sim")
