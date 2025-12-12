"""
Console/TUI testing utilities using pexpect and pyte.

Provides tools for testing interactive terminal applications:
- ConsoleTest: For TUI apps (like `uipath dev`) that redraw the screen
- PromptTest: For prompt-based CLIs (like `uipath debug`) with command/response

Usage (TUI):
    from testcases.common import ConsoleTest

    def test_my_tui():
        test = ConsoleTest(command="uv run uipath dev", test_name="my_test")
        try:
            test.start()
            test.wait_for_ui(3, "Initial load")
            test.send_key('r', "Run")
            test.expect_eof()
        finally:
            test.close()

Usage (Prompt-based):
    from testcases.common import PromptTest

    def test_my_cli():
        test = PromptTest(
            command="uv run uipath debug agent",
            test_name="my_test",
            prompt="> ",
        )
        try:
            test.start()
            test.send_command("help", expect="Available commands")
            test.send_command("quit")
            test.expect_eof()
        finally:
            test.close()
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pexpect
import pyte


# Default terminal dimensions
DEFAULT_COLS = 320
DEFAULT_ROWS = 60
DEFAULT_TIMEOUT = 60


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def read_log(filename: Union[str, Path], strip_codes: bool = True) -> str:
    """Read a log file, optionally stripping ANSI codes."""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    return strip_ansi(content) if strip_codes else content


@dataclass
class Frame:
    """A captured screen/output frame."""
    timestamp: float
    label: str
    content: str


class ConsoleTest:
    """Test harness for interactive TUI applications.

    Uses pyte terminal emulator to properly capture screen state for TUIs
    that redraw the entire screen (like textual/rich apps).
    Records frames during test execution and prints them all at the end.
    """

    def __init__(
        self,
        command: str,
        test_name: str,
        timeout: int = DEFAULT_TIMEOUT,
        cols: int = DEFAULT_COLS,
        rows: int = DEFAULT_ROWS,
        output_dir: Optional[Path] = None,
    ):
        self.command = command
        self.test_name = test_name
        self.timeout = timeout
        self.cols = cols
        self.rows = rows
        self.output_dir = output_dir or Path(".")

        self.child: Optional[pexpect.spawn] = None
        self.frames: list[Frame] = []
        self._log_handle = None
        self._log_path: Optional[Path] = None
        self._start_time: Optional[float] = None

        # pyte terminal emulator for proper screen rendering
        self._screen = pyte.Screen(cols, rows)
        self._stream = pyte.Stream(self._screen)

    def start(self):
        """Start the console application."""
        print(f"Starting: {self.command}")
        print(f"Test: {self.test_name}")

        self._start_time = time.time()
        self.frames = []
        self._screen.reset()

        self._log_path = self.output_dir / f"{self.test_name}.log"

        self.child = pexpect.spawn(
            self.command,
            encoding='utf-8',
            timeout=self.timeout,
            dimensions=(self.rows, self.cols),
        )

        self._log_handle = open(self._log_path, "w")
        self.child.logfile_read = self._log_handle

        time.sleep(2)
        self._read_and_feed()
        self._capture_frame("Initial UI")

    def _elapsed(self) -> float:
        if self._start_time:
            return time.time() - self._start_time
        return 0.0

    def _read_and_feed(self):
        """Read available output and feed to pyte terminal."""
        try:
            while True:
                try:
                    data = self.child.read_nonblocking(size=4096, timeout=0.1)
                    if data:
                        self._stream.feed(data)
                    else:
                        break
                except pexpect.TIMEOUT:
                    break
                except pexpect.EOF:
                    break
        except Exception:
            pass

    def _get_screen_content(self) -> str:
        """Get the current rendered screen content from pyte."""
        lines = []
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                char = self._screen.buffer[row][col]
                line += char.data if char.data else " "
            lines.append(line.rstrip())

        while lines and not lines[-1]:
            lines.pop()

        return "\n".join(lines)

    def _capture_frame(self, label: str):
        content = self._get_screen_content()
        self.frames.append(Frame(
            timestamp=self._elapsed(),
            label=label,
            content=content if content.strip() else "(empty screen)",
        ))

    def _print_frames(self):
        print(f"\n{'#'*80}")
        print(f"# RECORDING: {self.test_name}")
        print(f"# Frames: {len(self.frames)}")
        print(f"{'#'*80}")

        for i, frame in enumerate(self.frames):
            print(f"\n{'='*80}")
            print(f">>> Frame {i+1}/{len(self.frames)} [{frame.timestamp:.2f}s] {frame.label}")
            print('='*80)
            print(frame.content)

        print(f"\n{'#'*80}")
        print(f"# END RECORDING: {self.test_name}")
        print(f"{'#'*80}\n")

    def _save_frames(self):
        frames_path = self.output_dir / f"{self.test_name}_frames.txt"
        with open(frames_path, "w", encoding="utf-8") as f:
            f.write(f"RECORDING: {self.test_name}\n")
            f.write(f"Command: {self.command}\n")
            f.write(f"Frames: {len(self.frames)}\n")
            f.write("="*80 + "\n\n")

            for i, frame in enumerate(self.frames):
                f.write(f"--- Frame {i+1}/{len(self.frames)} [{frame.timestamp:.2f}s] {frame.label} ---\n")
                f.write("-"*80 + "\n")
                f.write(frame.content + "\n")
                f.write("\n")

        print(f"Frames saved to: {frames_path}")

    def close(self):
        """Close the test, print frames, and save to file."""
        self._read_and_feed()
        self._capture_frame("Final state")

        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

        if self.child:
            self.child.close()
            self.child = None

        self._print_frames()
        self._save_frames()
        print(f"Log saved to: {self._log_path}")

    def send_key(self, key: str, label: str = ""):
        self.child.send(key)
        time.sleep(0.5)
        self._read_and_feed()
        self._capture_frame(label or f"After key: {repr(key)}")

    def send_keys(self, keys: str, label: str = ""):
        self.child.send(keys)
        time.sleep(0.5)
        self._read_and_feed()
        self._capture_frame(label or f"After keys: {repr(keys)}")

    def send_line(self, line: str, label: str = ""):
        self.child.sendline(line)
        time.sleep(0.5)
        self._read_and_feed()
        self._capture_frame(label or f"After line: {line}")

    def expect(self, pattern: str, timeout: Optional[int] = None) -> int:
        result = self.child.expect(pattern, timeout=timeout or self.timeout)
        self._read_and_feed()
        self._capture_frame(f"Matched: {pattern}")
        return result

    def expect_any(self, patterns: list[str], timeout: Optional[int] = None) -> int:
        result = self.child.expect(patterns, timeout=timeout or self.timeout)
        self._read_and_feed()
        self._capture_frame(f"Matched pattern {result}")
        return result

    def expect_eof(self, timeout: Optional[int] = None):
        self.child.expect(pexpect.EOF, timeout=timeout or self.timeout)
        self._read_and_feed()
        self._capture_frame("Process exited (EOF)")

    def wait_for_ui(self, seconds: float = 1.0, label: str = ""):
        time.sleep(seconds)
        self._read_and_feed()
        self._capture_frame(label or f"Wait {seconds}s")

    def capture_screen(self, label: str):
        self._read_and_feed()
        self._capture_frame(label)

    def get_output(self, strip_codes: bool = True) -> str:
        if self._log_path and self._log_path.exists():
            if self._log_handle:
                self._log_handle.flush()
            return read_log(self._log_path, strip_codes)
        return ""

    @property
    def before(self) -> str:
        return self.child.before if self.child else ""


class PromptTest:
    """Test harness for prompt-based CLI applications.

    For CLIs that use a simple prompt (like `> `) and accept commands.
    Simpler than ConsoleTest - doesn't need pyte since output is sequential.
    Records interactions and prints them all at the end.
    """

    def __init__(
        self,
        command: str,
        test_name: str,
        prompt: str = "> ",
        timeout: int = DEFAULT_TIMEOUT,
        output_dir: Optional[Path] = None,
    ):
        self.command = command
        self.test_name = test_name
        self.prompt = prompt
        self.timeout = timeout
        self.output_dir = output_dir or Path(".")

        self.child: Optional[pexpect.spawn] = None
        self.frames: list[Frame] = []
        self._log_handle = None
        self._log_path: Optional[Path] = None
        self._start_time: Optional[float] = None

    def start(self):
        """Start the CLI application."""
        print(f"Starting: {self.command}")
        print(f"Test: {self.test_name}")

        self._start_time = time.time()
        self.frames = []

        self._log_path = self.output_dir / f"{self.test_name}.log"

        self.child = pexpect.spawn(
            self.command,
            encoding='utf-8',
            timeout=self.timeout,
        )

        self._log_handle = open(self._log_path, "w")
        self.child.logfile_read = self._log_handle

    def _elapsed(self) -> float:
        if self._start_time:
            return time.time() - self._start_time
        return 0.0

    def _capture_frame(self, label: str, content: str = ""):
        """Capture current output as a frame."""
        if not content and self.child:
            content = strip_ansi(self.child.before) if self.child.before else ""
        self.frames.append(Frame(
            timestamp=self._elapsed(),
            label=label,
            content=content if content.strip() else "(no output)",
        ))

    def _print_frames(self):
        print(f"\n{'#'*80}")
        print(f"# RECORDING: {self.test_name}")
        print(f"# Interactions: {len(self.frames)}")
        print(f"{'#'*80}")

        for i, frame in enumerate(self.frames):
            print(f"\n{'='*80}")
            print(f">>> [{frame.timestamp:.2f}s] {frame.label}")
            print('='*80)
            print(frame.content)

        print(f"\n{'#'*80}")
        print(f"# END RECORDING: {self.test_name}")
        print(f"{'#'*80}\n")

    def _save_frames(self):
        frames_path = self.output_dir / f"{self.test_name}_frames.txt"
        with open(frames_path, "w", encoding="utf-8") as f:
            f.write(f"RECORDING: {self.test_name}\n")
            f.write(f"Command: {self.command}\n")
            f.write(f"Interactions: {len(self.frames)}\n")
            f.write("="*80 + "\n\n")

            for i, frame in enumerate(self.frames):
                f.write(f"--- [{frame.timestamp:.2f}s] {frame.label} ---\n")
                f.write("-"*80 + "\n")
                f.write(frame.content + "\n")
                f.write("\n")

        print(f"Frames saved to: {frames_path}")

    def close(self):
        """Close the test, print frames, and save to file."""
        self._capture_frame("Final state")

        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

        if self.child:
            self.child.close()
            self.child = None

        self._print_frames()
        self._save_frames()
        print(f"Log saved to: {self._log_path}")

    def wait_for_prompt(self, label: str = ""):
        """Wait for the prompt to appear."""
        self.child.expect(self.prompt)
        self._capture_frame(label or "Prompt ready")

    def send_command(self, command: str, expect: Optional[str] = None, label: str = ""):
        """Send a command and optionally wait for expected output.

        Args:
            command: Command to send
            expect: Optional regex pattern to expect in response
            label: Description for the frame
        """
        self.child.expect(self.prompt)
        self.child.sendline(command)

        if expect:
            self.child.expect(expect)
            self._capture_frame(label or f"Command: {command} -> matched: {expect}")
        else:
            self._capture_frame(label or f"Command: {command}")

    def send_line(self, line: str, label: str = ""):
        """Send a line without waiting for prompt first."""
        self.child.sendline(line)
        self._capture_frame(label or f"Sent: {line}")

    def expect(self, pattern: str, timeout: Optional[int] = None) -> int:
        """Wait for a pattern in the output."""
        result = self.child.expect(pattern, timeout=timeout or self.timeout)
        self._capture_frame(f"Matched: {pattern}")
        return result

    def expect_eof(self, timeout: Optional[int] = None):
        """Wait for the process to exit."""
        self.child.expect(pexpect.EOF, timeout=timeout or self.timeout)
        self._capture_frame("Process exited (EOF)")

    def get_output(self, strip_codes: bool = True) -> str:
        """Get all output captured so far."""
        if self._log_path and self._log_path.exists():
            if self._log_handle:
                self._log_handle.flush()
            return read_log(self._log_path, strip_codes)
        return ""

    @property
    def before(self) -> str:
        """Get the output before the last expect match."""
        return self.child.before if self.child else ""
