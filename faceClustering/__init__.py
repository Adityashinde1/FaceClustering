from from_root import from_root
import logging
import os



logs_path = os.path.join(from_root(), faceClustering, artifacts, logs)
logging.basicConfig(filename=f"{logs_path}.txt",
                    format='[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)