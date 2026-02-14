"""
Animated spinner for terminal output.

Provides visual feedback during long-running operations
so the program doesn't appear stuck.

USAGE:
    # As a context manager (sync):
    with Spinner("Processing..."):
        do_something_slow()

    # As a context manager (async):
    async with Spinner("Thinking..."):
        await do_something_slow()

    # With custom style:
    with Spinner("Loading", style="dots"):
        do_something_slow()
"""

import sys
import threading
import time
import itertools


# Spinner animation frames
SPINNER_STYLES = {
    "default": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    "dots":    ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
    "line":    ["-", "\\", "|", "/"],
    "pulse":   ["◐", "◓", "◑", "◒"],
    "arrow":   ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
    "bounce":  ["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"],
}

# ANSI codes
CLEAR_LINE = "\033[2K"
CURSOR_UP = "\033[A"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CYAN = "\033[96m"
GRAY = "\033[90m"
RESET = "\033[0m"
BOLD = "\033[1m"


class Spinner:
    """
    Animated terminal spinner that runs in a background thread.

    Works as both a regular and async context manager.
    """

    def __init__(
        self,
        message: str = "Processing",
        style: str = "default",
        color: str = CYAN,
        speed: float = 0.08,
    ):
        self.message = message
        self.frames = itertools.cycle(SPINNER_STYLES.get(style, SPINNER_STYLES["default"]))
        self.color = color if sys.stdout.isatty() else ""
        self.reset = RESET if sys.stdout.isatty() else ""
        self.speed = speed
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._is_tty = sys.stdout.isatty()

    def _animate(self):
        """Background thread: render spinner frames."""
        if self._is_tty:
            sys.stdout.write(HIDE_CURSOR)
            sys.stdout.flush()

        while not self._stop_event.is_set():
            frame = next(self.frames)
            if self._is_tty:
                line = f"\r{self.color}{frame}{self.reset} {self.message}"
                sys.stdout.write(line)
                sys.stdout.flush()
            self._stop_event.wait(self.speed)

        # Clear the spinner line when done
        if self._is_tty:
            sys.stdout.write(f"\r{CLEAR_LINE}")
            sys.stdout.write(SHOW_CURSOR)
            sys.stdout.flush()

    def start(self):
        """Start the spinner animation."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the spinner animation."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    # Sync context manager
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # Async context manager
    async def __aenter__(self):
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
