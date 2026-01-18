import logging
import os

class Logger:
    def __init__(self, log_dir: str = "logs", filename: str = "app.log") -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = os.path.abspath(os.path.join(self.log_dir, filename))

        # Use a logger name that is unique per output file
        self.logger = logging.getLogger(f"app:{self.log_path}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # don't forward to root logger

        # Avoid duplicate handlers if Logger() is constructed multiple times
        if not self.logger.handlers:
            fmt = logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            file_handler = logging.FileHandler(self.log_path, encoding="utf-8", mode="a")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(fmt)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(fmt)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _flush(self) -> None:
        for h in self.logger.handlers:
            try:
                h.flush()
            except Exception:
                pass

    def log(self, message: str) -> None:
        self.logger.info(message)
        self._flush()

    def warn(self, message: str) -> None:
        self.logger.warning(message)
        self._flush()

    def error(self, message: str) -> None:
        self.logger.error(message)
        self._flush()