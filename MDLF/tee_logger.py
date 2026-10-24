"""
tee_logger.py
=============
Utilidad para duplicar stdout y logging a un fichero de log.
"""

import logging
import sys
from pathlib import Path


class Tee:
    """
    Context manager que escribe simultáneamente en stdout y en un fichero.

    También añade un handler al logger raíz para que los mensajes de
    logging.info/warning/etc. queden registrados en el mismo fichero.

    Uso:
        with Tee(Path("runs/run_20250504.log")):
            print("esto va a stdout Y al fichero")
            logging.info("esto también")
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, msg: str) -> None:
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._stdout
        self._file.close()

    def __enter__(self):
        sys.stdout = self
        self._log_handler = logging.StreamHandler(self._file)
        self._log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logging.getLogger().addHandler(self._log_handler)
        return self

    def __exit__(self, *_):
        logging.getLogger().removeHandler(self._log_handler)
        self.close()
