# path: app/NetNaviApp_refactor.py
"""NetNavi Assistant with Dynamic Emotional Core
- GUI if PyQt5 is available; otherwise headless CLI fallback.
- OpenAI chat, optional voice (SR optional), instant search, tasks DB.
- EmotionEngine influences UI theme/glow and TTS parameters.
- Unit tests runnable via env var: NETNAVI_RUN_TESTS=1
- Headless non-interactive safe: avoid stdin when sandboxed; use --once or env.

CLI flags/env:
  --once "<cmd>"     Process a single command and exit (e.g., --once "list tasks").
  --no-tts           Disable TTS regardless of availability.
  NETNAVI_NONINTERACTIVE=1   Force non-interactive CLI (no input()).
  NETNAVI_DISABLE_TTS=1      Disable TTS.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import sqlite3
import time
import re
import math
import tempfile
import unittest
from typing import List, Tuple, Optional, Dict

# --- requests optional (fallback to urllib) ---
try:  # pragma: no cover - env dependent
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None  # type: ignore
    REQUESTS_AVAILABLE = False
    import urllib.request
    import urllib.parse
    import json as _json

try:  # optional macOS notifier
    import pync  # type: ignore
except Exception:  # pragma: no cover
    pync = None

# --- pyttsx3 TTS optional and safe ---
try:  # pragma: no cover - env dependent
    import pyttsx3  # type: ignore
    TTS_AVAILABLE = True
except Exception:
    pyttsx3 = None  # type: ignore
    TTS_AVAILABLE = False

# --- speech_recognition is optional ---
try:  # pragma: no cover - env dependent
    import speech_recognition as sr  # type: ignore
    SR_AVAILABLE = True
except Exception:
    sr = None  # type: ignore
    SR_AVAILABLE = False

# --- Qt bindings optional (PyQt5 → PyQt6 → PySide6) ---
QT_LIB = None
try:  # Try PyQt5 first
    from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QEvent
    from PyQt5.QtGui import QPixmap, QColor
    from PyQt5.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QInputDialog,
        QGraphicsDropShadowEffect,
        QDialog,
        QCheckBox,
        QDialogButtonBox,
        QAction,
    )
    GUI_AVAILABLE = True
    QT_LIB = "PyQt5"
except Exception:
    try:  # Fallback: PyQt6
        from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QEvent
        from PyQt6.QtGui import QPixmap, QColor, QAction
        from PyQt6.QtWidgets import (
            QApplication,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
            QInputDialog,
            QGraphicsDropShadowEffect,
            QDialog,
            QCheckBox,
            QDialogButtonBox,
        )
        GUI_AVAILABLE = True
        QT_LIB = "PyQt6"
    except Exception:
        try:  # Fallback: PySide6
            from PySide6.QtCore import Qt, Signal as pyqtSignal, QObject, QThread, QEvent
            from PySide6.QtGui import QPixmap, QColor, QAction
            from PySide6.QtWidgets import (
                QApplication,
                QHBoxLayout,
                QLabel,
                QLineEdit,
                QMainWindow,
                QMessageBox,
                QPushButton,
                QTextEdit,
                QVBoxLayout,
                QWidget,
                QInputDialog,
                QGraphicsDropShadowEffect,
                QDialog,
                QCheckBox,
                QDialogButtonBox,
            )
            GUI_AVAILABLE = True
            QT_LIB = "PySide6"
        except Exception:
            GUI_AVAILABLE = False
            QT_LIB = None
            # Placeholders to keep type hints happy at runtime
            Qt = object  # type: ignore
            pyqtSignal = lambda *_, **__: None  # type: ignore
            QObject = object  # type: ignore
            QThread = object  # type: ignore
            QEvent = object  # type: ignore
            QPixmap = object  # type: ignore
            def QColor(*_args, **_kwargs):  # type: ignore
                return None

# Qt compat: enum aliases + app exec helper
if GUI_AVAILABLE:
    try:  # PyQt6/PySide6
        KeepAspect = Qt.AspectRatioMode.KeepAspectRatio  # type: ignore[attr-defined]
    except Exception:  # PyQt5
        KeepAspect = getattr(Qt, "KeepAspectRatio", None)
    try:  # PyQt6/PySide6
        SmoothTransform = Qt.TransformationMode.SmoothTransformation  # type: ignore[attr-defined]
    except Exception:
        SmoothTransform = getattr(Qt, "SmoothTransformation", None)

    def qt_exec(app: "QApplication") -> int:  # pragma: no cover
        fn = getattr(app, "exec", None)
        if callable(fn):
            return fn()
        return app.exec_()  # type: ignore[attr-defined]

# --- OpenAI SDK optional ---
try:  # pragma: no cover - env dependent
    from openai import OpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

# ---------------- CONSTANTS ---------------- #
CONFIG_FILE = "config.json"
DB_FILE = "tasks.db"
DEFAULT_CONFIG = {
    "api_key": "",
    "avatar": "assets/avatars/default.png",
    "theme": {"bg": "#1e1e1e", "fg": "#00ffcc"},
    "show_voice_control_if_unavailable": True,
    "emotion_enabled": True,
}

# ---------------- CONFIG ---------------- #

def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        os.makedirs(os.path.dirname(CONFIG_FILE) or ".", exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


config = load_config()
CLIENT: Optional["OpenAI"] = None
if OPENAI_AVAILABLE and config.get("api_key"):
    try:
        CLIENT = OpenAI(api_key=config["api_key"])  # type: ignore[name-defined]
    except Exception:
        CLIENT = None

# ---------------- UTIL: NOTIFY ---------------- #

def notify(title: str, message: str) -> None:
    """Best-effort notification in any environment."""
    try:
        if pync:
            pync.notify(message, title=title)
        else:
            print(f"[{title}] {message}")
    except Exception:
        print(f"[{title}] {message}")


# ---------------- VOICE: TTS + STT ---------------- #
_tts_lock = threading.Lock()
_engine = None

# Attempt to init TTS engine, but fully disable on failure (e.g., missing eSpeak)
if TTS_AVAILABLE:
    try:
        _engine = pyttsx3.init()  # type: ignore[name-defined]
    except Exception as _tts_err:  # pragma: no cover - platform dependent
        print(f"[TTS disabled] init failed: {_tts_err}")
        TTS_AVAILABLE = False
        _engine = None


def speak(text: str, rate: Optional[int] = None, volume: Optional[float] = None) -> None:
    # Allow explicit disable in CI/sandbox to avoid audio deps
    if os.environ.get("NETNAVI_DISABLE_TTS") == "1" or not TTS_AVAILABLE or _engine is None:
        print(f"[TTS disabled] {text}")
        return
    with _tts_lock:
        try:
            if rate is not None:
                _engine.setProperty("rate", int(rate))  # type: ignore[attr-defined]
            if volume is not None:
                _engine.setProperty("volume", max(0.0, min(1.0, float(volume))))  # type: ignore[attr-defined]
            _engine.say(text)  # type: ignore[attr-defined]
            _engine.runAndWait()  # type: ignore[attr-defined]
        except Exception as e:  # Last-resort fallback
            print(f"[TTS disabled after error] {e}: {text}")


def speak_async(text: str, rate: Optional[int] = None, volume: Optional[float] = None) -> None:
    if os.environ.get("NETNAVI_DISABLE_TTS") == "1" or not TTS_AVAILABLE or _engine is None:
        print(f"[TTS disabled] {text}")
        return
    threading.Thread(target=speak, args=(text, rate, volume), daemon=True).start()


def listen_voice(timeout: int = 5, phrase_time_limit: int = 10) -> str:
    """Return recognized text or '' if voice is unavailable (non-blocking in tests)."""
    if os.environ.get("NETNAVI_RUN_TESTS") == "1":
        return ""
    if not SR_AVAILABLE:
        return ""
    try:  # pragma: no cover - device dependent
        recognizer = sr.Recognizer()  # type: ignore
        with sr.Microphone() as source:  # type: ignore
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        return recognizer.recognize_google(audio)
    except Exception:
        return ""


# ---------------- EMOTIONAL CORE ---------------- #
class EmotionEngine:
    """Tiny valence/arousal state with decay + heuristics."""

    POSITIVE = {
        "thanks", "thank", "great", "awesome", "nice", "love", "good", "cool", "amazing", "wow",
        "yay", "success", "happy", "joy", "perfect", "excellent", "win", "sweet",
    }
    NEGATIVE = {
        "error", "fail", "broken", "hate", "bad", "angry", "annoy", "wtf", "issue", "bug",
        "crash", "slow", "sad", "upset", "stupid", "dumb", "terrible", "ugh", "oops",
    }
    AROUSAL_HIGH = {"urgent", "now", "immediately", "!", "help", "asap"}

    THEMES = {
        "joy": {"bg": "#112b2b", "fg": "#00ffc8"},
        "calm": {"bg": "#1e2430", "fg": "#c8e1ff"},
        "curious": {"bg": "#201a2d", "fg": "#cdb4ff"},
        "focused": {"bg": "#161a1f", "fg": "#b8d4ff"},
        "frustrated": {"bg": "#2a1414", "fg": "#ffb3b3"},
        "tired": {"bg": "#1a1a1a", "fg": "#aaaaaa"},
    }
    EMOJIS = {
        "joy": "😄",
        "calm": "🙂",
        "curious": "🧐",
        "focused": "🤖",
        "frustrated": "😤",
        "tired": "😴",
    }

    def __init__(self) -> None:
        self.valence = 0.0
        self.arousal = 0.3
        self.last_ts = time.time()
        self._lock = threading.Lock()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z']+", text.lower())

    def _score_text(self, text: str) -> Tuple[float, float]:
        toks = self._tokenize(text)
        if not toks:
            return 0.0, 0.0
        pos = sum(1 for t in toks if t in self.POSITIVE)
        neg = sum(1 for t in toks if t in self.NEGATIVE)
        val_raw = (pos - neg) / max(1, len(toks))
        excite = 0.0
        excite += min(text.count("!"), 3) * 0.1
        excite += 0.15 if any(t in toks for t in self.AROUSAL_HIGH) else 0.0
        alpha = sum(1 for c in text if c.isalpha())
        caps_ratio = (sum(1 for c in text if c.isupper()) / alpha) if alpha else 0.0
        excite += 0.15 if caps_ratio > 0.6 and len(text) >= 8 else 0.0
        excite = min(0.7, excite)
        return float(val_raw), float(excite)

    def _decay(self) -> None:
        now = time.time()
        dt = now - self.last_ts
        self.last_ts = now
        decay = math.exp(-dt / 30.0)
        self.valence *= decay
        self.arousal = 0.2 + (self.arousal - 0.2) * decay

    def update_from_text(self, text: str, source: str = "user") -> None:
        with self._lock:
            self._decay()
            dv, da = self._score_text(text)
            w = 0.9 if source == "user" else 0.5
            self.valence = max(-1.0, min(1.0, self.valence + w * dv))
            self.arousal = max(0.0, min(1.0, self.arousal + w * da))

    def mood(self) -> str:
        v, a = self.valence, self.arousal
        if a < 0.25:
            return "tired"
        if v < -0.3:
            return "frustrated"
        if v > 0.4 and a > 0.5:
            return "joy"
        if 0.2 < v and a < 0.5:
            return "calm"
        if 0.4 <= a <= 0.7 and abs(v) < 0.2:
            return "curious"
        return "focused"

    def theme(self) -> Dict[str, str]:
        return self.THEMES[self.mood()]

    def emoji(self) -> str:
        return self.EMOJIS[self.mood()]

    def glow(self) -> str:
        """Return a hex color string. GUI converts to QColor if available."""
        mood = self.mood()
        if mood == "joy":
            return "#00ffc8"
        if mood == "calm":
            return "#6aa9ff"
        if mood == "curious":
            return "#b088ff"
        if mood == "frustrated":
            return "#ff6b6b"
        if mood == "tired":
            return "#888888"
        return "#9ad0ff"

    def tts_params(self) -> Tuple[int, float]:
        rate = int(160 + self.arousal * 80)
        vol = max(0.5, min(1.0, 0.7 + 0.15 * self.arousal + 0.1 * max(0.0, self.valence)))
        if self.mood() == "tired":
            rate = max(140, rate - 20)
        if self.mood() == "frustrated":
            rate = min(240, rate + 20)
        return rate, vol

    def mood_system_prompt(self) -> str:
        return (
            f"Assistant internal mood: {self.mood()} (valence={self.valence:.2f}, arousal={self.arousal:.2f}). "
            "Be helpful and professional; subtly reflect this mood in warmth/conciseness without overdoing it."
        )


EMOTION = EmotionEngine()


# ---------------- SEARCH ---------------- #

def _http_get_json(url: str, params: Dict[str, str], timeout: int = 10) -> Dict[str, object]:
    """Requests if available; fallback to urllib. Avoids adding deps in sandbox."""
    if REQUESTS_AVAILABLE:
        try:
            resp = requests.get(url, params=params, timeout=timeout)  # type: ignore[name-defined]
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]
        except Exception as e:
            return {"_error": str(e)}
    # urllib fallback
    try:
        q = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f"{url}?{q}", timeout=timeout) as r:  # type: ignore[attr-defined]
            data = r.read().decode("utf-8", errors="replace")
            return _json.loads(data)
    except Exception as e:  # pragma: no cover
        return {"_error": str(e)}


def search_query(query: str) -> str:
    data = _http_get_json(
        "https://api.duckduckgo.com/",
        {"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
        timeout=10,
    )
    if "_error" in data:
        return f"[Search error: {data['_error']}]"
    return str(data.get("AbstractText") or data.get("Heading") or "No summary available.")

# ---------------- TASKS (SQLite) ---------------- #

def init_db() -> None:
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tasks ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  description TEXT NOT NULL,"
            "  due TEXT DEFAULT ''"
            ")"
        )


def add_task(desc: str, due: str = "") -> None:
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("INSERT INTO tasks (description, due) VALUES (?, ?)", (desc, due))
    notify("NetNavi Task", f"Task added: {desc}")


def list_tasks() -> List[Tuple[int, str, str]]:
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.execute("SELECT id, description, due FROM tasks ORDER BY id ASC")
        return list(cur.fetchall())


def delete_task(task_id: int) -> bool:
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cur.rowcount > 0


# ---------------- AI ENGINE ---------------- #

def chat_with_ai(user_message: str, mood_prompt: str) -> str:
    global CLIENT
    if CLIENT is None:
        return "[OpenAI error: missing/invalid API key. Set it in config.json or via the dialog.]"
    try:
        resp = CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly NetNavi assistant."},
                {"role": "system", "content": mood_prompt},
                {"role": "user", "content": user_message},
            ],
            timeout=60,
        )
        return resp.choices[0].message.content or ""
    except TypeError:  # SDK differences
        try:
            resp = CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a friendly NetNavi assistant."},
                    {"role": "system", "content": mood_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as inner:
            return f"[Error contacting AI: {inner}]"
    except Exception as e:
        msg = str(e)
        if "api_key" in msg.lower() or "authentication" in msg.lower():
            return "[OpenAI error: missing/invalid API key. Set it in config.json or via the dialog.]"
        return f"[Error contacting AI: {e}]"


# ---------------- SHARED MESSAGE PROCESSOR ---------------- #

def process_message_sync(msg: str, mood_prompt: str) -> str:
    """Process a user message and return a response (shared by GUI/CLI)."""
    m = msg.strip()
    lower = m.lower()

    if lower.startswith("search:"):
        query = m.split(":", 1)[1].strip()
        return search_query(query)

    if lower.startswith("task:"):
        task_desc = m.split(":", 1)[1].strip()
        add_task(task_desc)
        return f"Task added: {task_desc}"

    if lower.startswith("delete task"):
        parts = lower.split()
        if len(parts) >= 3 and parts[2].isdigit():
            ok = delete_task(int(parts[2]))
            return "Task deleted." if ok else "Task not found."
        return "Usage: delete task <id>"

    if lower == "list tasks":
        tasks = list_tasks()
        if tasks:
            return "\n".join(f"{tid}: {desc} (due {due or '—'})" for tid, desc, due in tasks)
        return "No tasks yet."

    # Default: AI
    return chat_with_ai(m, mood_prompt)


# ---------------- GUI ---------------- #
if GUI_AVAILABLE:

    class Worker(QObject):
        finished = pyqtSignal(str)
        error = pyqtSignal(str)

        def __init__(self, message: str, mood_prompt: str):
            super().__init__()
            self.message = message
            self.mood_prompt = mood_prompt

        def run(self) -> None:
            try:
                reply = process_message_sync(self.message, self.mood_prompt)
                self.finished.emit(reply)
            except Exception as e:  # defensive
                self.error.emit(str(e))

    class NetNaviApp(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("NetNavi Assistant")
            self.setGeometry(100, 100, 700, 560)
            self.last_msg: Optional[str] = None
            self._voice_hint_shown = False

            # Avatar
            self.avatar = QLabel()
            if os.path.exists(config.get("avatar", "")):
                pixmap = QPixmap(config["avatar"]).scaled(150, 150, KeepAspect, SmoothTransform)
                self.avatar.setPixmap(pixmap)
            self.avatar.setAlignment(Qt.AlignCenter)
            self._glow = QGraphicsDropShadowEffect(self)
            self._glow.setOffset(0, 0)
            self._glow.setBlurRadius(40)
            self.avatar.setGraphicsEffect(self._glow)

            # Chat log
            self.chat_log = QTextEdit(); self.chat_log.setReadOnly(True)

            # Input
            self.input_box = QLineEdit(); self.input_box.setPlaceholderText("Type a message…")
            self.input_box.returnPressed.connect(self.handle_send)

            # Buttons
            self.send_btn = QPushButton("Send"); self.send_btn.clicked.connect(self.handle_send)
            self.voice_btn = QPushButton("🎤 Speak"); self.voice_btn.clicked.connect(self.handle_voice)
            if not SR_AVAILABLE:
                if config.get("show_voice_control_if_unavailable", True):
                    self.voice_btn.setToolTip("Voice input unavailable (module not installed)")
                else:
                    self.voice_btn.hide()
            self.settings_btn = QPushButton("Settings"); self.settings_btn.clicked.connect(self.open_settings)

            btn_layout = QHBoxLayout()
            btn_layout.addWidget(self.send_btn)
            btn_layout.addWidget(self.voice_btn)
            btn_layout.addWidget(self.settings_btn)
            self.status_label = QLabel("● API: unknown"); self.status_label.setToolTip("OpenAI API connectivity status")
            self.retry_btn = QPushButton("Retry last"); self.retry_btn.setEnabled(False); self.retry_btn.clicked.connect(self.retry_last)
            self.check_btn = QPushButton("API Status"); self.check_btn.clicked.connect(self.check_api_status)
            btn_layout.addStretch(); btn_layout.addWidget(self.check_btn); btn_layout.addWidget(self.retry_btn)

            status_layout = QHBoxLayout(); status_layout.addWidget(self.status_label); status_layout.addStretch()

            layout = QVBoxLayout()
            layout.addWidget(self.avatar)
            layout.addWidget(self.chat_log)
            layout.addWidget(self.input_box)
            layout.addLayout(btn_layout)
            layout.addLayout(status_layout)
            container = QWidget(); container.setLayout(layout); self.setCentralWidget(container)

            # Menu bar
            self.build_menu()

            # Initial theme + welcome
            self.apply_emotion_ui()
            if CLIENT is None:
                self.append_chat("Navi", "AI is <b>Offline</b> (no API key or SDK). Tasks & search work; add an API key in Settings to enable AI.")
            else:
                self.append_chat("Navi", "Ready. Try: search: quantum computing | task: Buy milk | list tasks")

        def build_menu(self) -> None:
            mb = self.menuBar()
            file_menu = mb.addMenu("File")
            act_settings = QAction("Settings", self); act_settings.triggered.connect(self.open_settings)
            file_menu.addAction(act_settings)
            help_menu = mb.addMenu("Help")
            act_about = QAction("About", self); act_about.triggered.connect(self.show_about)
            help_menu.addAction(act_about)

        # ----- UI helpers ----- #
        def apply_emotion_ui(self) -> None:
            if not config.get("emotion_enabled", True):
                theme = config.get("theme", {"bg": "#1e1e1e", "fg": "#00ffcc"})
                self.setStyleSheet(f"background-color: {theme['bg']}; color: {theme['fg']};")
                self._glow.setColor(QColor("#000000"))
                self._glow.setBlurRadius(0)
                return
            theme = EMOTION.theme()
            self.setStyleSheet(f"background-color: {theme['bg']}; color: {theme['fg']};")
            self._glow.setColor(QColor(EMOTION.glow()))
            self._glow.setBlurRadius(25 + int(25 * EMOTION.arousal))

        def append_chat(self, who: str, text: str) -> None:
            label = f"{EMOTION.emoji()} {who}" if who == "Navi" else who
            self.chat_log.append(f"<b>{label}:</b> {text}")

        # ----- Actions ----- #
        def handle_send(self) -> None:
            msg = self.input_box.text().strip()
            if not msg:
                return
            self.input_box.clear(); self.last_msg = msg
            self.append_chat("You", msg)
            EMOTION.update_from_text(msg, source="user"); self.apply_emotion_ui()
            self.run_worker(msg)

        def handle_voice(self) -> None:
            if not SR_AVAILABLE:
                if not self._voice_hint_shown and config.get("show_voice_control_if_unavailable", True):
                    self._voice_hint_shown = True
                    hint = (
                        "Voice input isn't available here. To enable it locally, install dependencies: "
                        "`pip install SpeechRecognition pyaudio` (on macOS also `brew install portaudio`)."
                    )
                    self.append_chat("Navi", hint)
                return
            self.append_chat("Navi", "Listening…")

            def _capture():
                text = listen_voice()
                if text:
                    QApplication.postEvent(self, _CallableEvent(lambda: self.append_chat("You (voice)", text)))
                    EMOTION.update_from_text(text, source="user")
                    QApplication.postEvent(self, _CallableEvent(self.apply_emotion_ui))
                    self.run_worker(text)
                else:
                    QApplication.postEvent(self, _CallableEvent(lambda: self.append_chat("Navi", "Sorry, I didn't catch that.")))

            threading.Thread(target=_capture, daemon=True).start()

        def run_worker(self, msg: str) -> None:
            self.thread = QThread(self)
            self.worker = Worker(msg, EMOTION.mood_system_prompt())
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.error.connect(self.on_worker_error)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()

        def on_worker_finished(self, text: str) -> None:
            EMOTION.update_from_text(text, source="navi"); self.apply_emotion_ui()
            self.append_chat("Navi", text)
            api_ok = not (text.startswith("[") and "OpenAI" in text)
            self.set_status(api_ok, "OK" if api_ok else "Error")
            if text and not text.startswith("["):
                rate, vol = EMOTION.tts_params(); speak_async(text[:400], rate=rate, volume=vol)
            self.retry_btn.setEnabled(self.last_msg is not None)

        def on_worker_error(self, err: str) -> None:
            self.append_chat("Navi", f"[Error: {err}]")
            self.set_status(False, err)
            self.retry_btn.setEnabled(self.last_msg is not None)

        def retry_last(self) -> None:
            if self.last_msg:
                self.run_worker(self.last_msg)

        def set_status(self, ok: bool, msg: str = "") -> None:
            color = "#2ecc71" if ok else "#e74c3c"
            self.status_label.setText(f"<span style='color:{color}'>●</span> API: {msg}")

        def check_api_status(self) -> None:
            if CLIENT is None:
                self.set_status(False, "Offline"); return
            ok, info = api_healthcheck(); self.set_status(ok, info)

        def show_about(self) -> None:
            QMessageBox.about(
                self,
                "About NetNavi Assistant",
                (
                    "<b>NetNavi Assistant</b><br>"
                    "Chat UI with optional voice, tasks DB, instant search, and an emotional core."\
                    "<br><br><b>Voice setup</b>:<br>pip install SpeechRecognition pyaudio<br>macOS: brew install portaudio<br>"\
                    "<br><b>AI setup</b>: Add your OpenAI API key in Settings."
                ),
            )

        def open_settings(self) -> None:
            changed = open_settings_dialog(self)
            if changed:
                if not SR_AVAILABLE:
                    if config.get("show_voice_control_if_unavailable", True):
                        self.voice_btn.show(); self.voice_btn.setToolTip("Voice input unavailable (module not installed)")
                    else:
                        self.voice_btn.hide()
                self.apply_emotion_ui()

        def event(self, e: QEvent):  # type: ignore[override]
            if isinstance(e, _CallableEvent):
                try:
                    e.fn()
                finally:
                    return True
            return super().event(e)

    class _CallableEvent(QEvent):
        _etype = QEvent.Type(QEvent.registerEventType())
        def __init__(self, fn):
            super().__init__(self._etype); self.fn = fn

    def open_settings_dialog(parent: QWidget) -> bool:
        dlg = QDialog(parent); dlg.setWindowTitle("Settings")
        layout = QVBoxLayout(dlg)
        chk_voice = QCheckBox("Show voice controls when unavailable"); chk_voice.setChecked(config.get("show_voice_control_if_unavailable", True))
        chk_emotion = QCheckBox("Enable emotional core"); chk_emotion.setChecked(config.get("emotion_enabled", True))
        layout.addWidget(chk_voice); layout.addWidget(chk_emotion)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel); layout.addWidget(buttons)

        def on_save():
            config["show_voice_control_if_unavailable"] = bool(chk_voice.isChecked())
            config["emotion_enabled"] = bool(chk_emotion.isChecked())
            save_config(config); dlg.accept()

        buttons.accepted.connect(on_save); buttons.rejected.connect(dlg.reject)
        return dlg.exec_() == QDialog.Accepted


# ---------------- TESTS ---------------- #

def run_tests() -> None:
    class DBTests(unittest.TestCase):
        def setUp(self):
            self.tmpdir = tempfile.TemporaryDirectory()
            self.old_db = globals()["DB_FILE"]
            globals()["DB_FILE"] = os.path.join(self.tmpdir.name, "test_tasks.db")
            init_db()
        def tearDown(self):
            globals()["DB_FILE"] = self.old_db; self.tmpdir.cleanup()
        def test_init_and_list_empty(self):
            self.assertEqual(list_tasks(), [])
        def test_add_and_list_and_delete(self):
            add_task("Buy milk", "2025-12-31"); tasks = list_tasks(); self.assertEqual(len(tasks), 1)
            tid, desc, due = tasks[0]; self.assertEqual(desc, "Buy milk"); self.assertEqual(due, "2025-12-31")
            self.assertTrue(delete_task(tid)); self.assertEqual(list_tasks(), [])
        def test_delete_missing(self):
            self.assertFalse(delete_task(999999))

    class EmotionTests(unittest.TestCase):
        def test_frustrated_and_joy(self):
            e = EmotionEngine(); e.update_from_text("this is broken and dumb!", source="user")
            self.assertIn(e.mood(), {"frustrated", "focused"})
            e.update_from_text("thanks, this is great!", source="user")
            self.assertIn(e.mood(), {"joy", "calm", "focused"})
        def test_tts_params_range(self):
            e = EmotionEngine(); e.arousal = 0.0; r, v = e.tts_params(); self.assertTrue(120 <= r <= 240); self.assertTrue(0.5 <= v <= 1.0)

    class AITests(unittest.TestCase):
        def test_ai_when_no_client(self):
            global CLIENT; old = CLIENT; CLIENT = None
            msg = chat_with_ai("hello", "mood"); self.assertIn("OpenAI error", msg); CLIENT = old
        def test_offline_mode_status_string(self):
            global CLIENT; old = CLIENT; CLIENT = None
            ok, info = api_healthcheck(); self.assertFalse(ok); self.assertIn("No API key", info); CLIENT = old

    class VoiceTests(unittest.TestCase):
        def test_listen_voice_returns_string(self):
            os.environ["NETNAVI_RUN_TESTS"] = "1"; txt = listen_voice(); self.assertIsInstance(txt, str); self.assertEqual(txt, ""); os.environ.pop("NETNAVI_RUN_TESTS", None)
        def test_default_config_voice_toggle_present(self):
            self.assertIn("show_voice_control_if_unavailable", DEFAULT_CONFIG); self.assertTrue(DEFAULT_CONFIG["show_voice_control_if_unavailable"])

    class EnvTests(unittest.TestCase):
        def test_gui_available_flag_present(self):
            self.assertIn("GUI_AVAILABLE", globals()); self.assertIsInstance(globals()["GUI_AVAILABLE"], bool)
        def test_openai_available_flag_present(self):
            self.assertIn("OPENAI_AVAILABLE", globals()); self.assertIsInstance(globals()["OPENAI_AVAILABLE"], bool)
        def test_tts_available_flag_present(self):
            self.assertIn("TTS_AVAILABLE", globals()); self.assertIsInstance(globals()["TTS_AVAILABLE"], bool)
        def test_requests_available_flag_present(self):
            self.assertIn("REQUESTS_AVAILABLE", globals()); self.assertIsInstance(globals()["REQUESTS_AVAILABLE"], bool)

    class TTSTests(unittest.TestCase):
        def test_speak_no_crash_when_disabled(self):
            os.environ["NETNAVI_DISABLE_TTS"] = "1"
            # Should not raise even if pyttsx3 is present/missing driver
            speak("Hello test")
            speak_async("Hello async")
            os.environ.pop("NETNAVI_DISABLE_TTS", None)

    class SearchTests(unittest.TestCase):
        def test_search_query_returns_string(self):
            out = search_query("test"); self.assertIsInstance(out, str); self.assertGreaterEqual(len(out), 0)

    class CLITests(unittest.TestCase):
        def test_cli_noninteractive_env_returns(self):
            os.environ["NETNAVI_NONINTERACTIVE"] = "1"
            # Ensure no exceptions when non-interactive
            cli_main([])
            os.environ.pop("NETNAVI_NONINTERACTIVE", None)
        def test_cli_once_command(self):
            # Should process a single command and exit without reading stdin
            cli_main(["--once", "list tasks", "--no-tts"])  # no assertion; just ensure no crash

    suite = unittest.TestSuite()
    for case in (DBTests, EmotionTests, AITests, VoiceTests, EnvTests, TTSTests, SearchTests, CLITests):
        suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


# ---------------- HEALTHCHECK ---------------- #

def api_healthcheck() -> Tuple[bool, str]:
    global CLIENT
    if CLIENT is None:
        # Keep message stable for tests even if SDK is missing
        return False, "No API key"
    try:
        _ = CLIENT.models.list()
        return True, "OK"
    except Exception as e:
        return False, str(e)

# ---------------- MAIN ---------------- #

def ensure_api_key_qt() -> None:
    """GUI path: ask for key, or run Offline Mode (also if SDK missing).
    Headless envs should use ensure_api_key_headless().
    """
    global CLIENT
    if CLIENT is not None:
        return
    if not OPENAI_AVAILABLE:
        try:
            QMessageBox.information(None, "NetNavi Assistant", "OpenAI SDK not available. Running in Offline Mode.")  # type: ignore[name-defined]
        except Exception:
            pass
        CLIENT = None
        return
    api_key = config.get("api_key", "").strip()
    if not api_key:
        api_key_input, ok = QInputDialog.getText(None, "OpenAI API Key", "Enter your OpenAI API Key:")  # type: ignore[name-defined]
        if not ok or not api_key_input.strip():
            QMessageBox.information(None, "NetNavi Assistant", "Launching in Offline Mode (AI disabled).")  # type: ignore[name-defined]
            CLIENT = None; return
        api_key = api_key_input.strip(); config["api_key"] = api_key; save_config(config)
    try:
        CLIENT = OpenAI(api_key=api_key)  # type: ignore[name-defined]
    except Exception as e:
        try:
            QMessageBox.warning(None, "NetNavi Assistant", f"Could not init OpenAI client. Running Offline.\n{e}")  # type: ignore[name-defined]
        except Exception:
            pass
        CLIENT = None


def ensure_api_key_headless() -> None:
    """Headless path: no prompts; stay offline if key missing/invalid or SDK missing."""
    global CLIENT
    if CLIENT is not None:
        return
    if not OPENAI_AVAILABLE:
        print("[NetNavi] OpenAI SDK not available. Offline Mode.")
        CLIENT = None; return
    api_key = config.get("api_key", "").strip()
    if not api_key:
        print("[NetNavi] Offline Mode (no API key). Tasks & search still work.")
        CLIENT = None; return
    try:
        CLIENT = OpenAI(api_key=api_key)  # type: ignore[name-defined]
    except Exception as e:
        print(f"[NetNavi] OpenAI init failed; staying Offline: {e}")
        CLIENT = None


def cli_main(argv: Optional[List[str]] = None) -> None:
    """Headless CLI.
    - Non-interactive safe: if stdin is not a TTY or NETNAVI_NONINTERACTIVE=1, do not call input().
    - Supports --once "<cmd>" to run a single command without prompting.
    - Supports --no-tts to disable speech.
    """
    argv = argv or sys.argv[1:]

    # Simple flag parsing (no argparse to keep deps minimal)
    once_cmd: Optional[str] = None
    disable_tts = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--once" and i + 1 < len(argv):
            once_cmd = argv[i + 1]
            i += 2; continue
        if a == "--no-tts":
            disable_tts = True
            i += 1; continue
        i += 1

    if disable_tts:
        os.environ["NETNAVI_DISABLE_TTS"] = "1"

    ensure_api_key_headless()
    init_db()

    # One-shot command mode
    if once_cmd:
        EMOTION.update_from_text(once_cmd, source="user")
        reply = process_message_sync(once_cmd, EMOTION.mood_system_prompt())
        EMOTION.update_from_text(reply, source="navi")
        print(f"Navi: {reply}")
        if reply and not reply.startswith("["):
            rate, vol = EMOTION.tts_params(); speak_async(reply[:400], rate=rate, volume=vol)
        return

    # Non-interactive detection
    non_interactive_env = os.environ.get("NETNAVI_NONINTERACTIVE") == "1"
    is_tty = False
    try:
        is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
    except Exception:
        is_tty = False

    if non_interactive_env or not is_tty:
        print("[NetNavi] Non-interactive environment detected; CLI loop disabled. Use --once \"<cmd>\" to run a single command.")
        return

    # Interactive loop
    print("NetNavi (headless). Type 'exit' to quit. Examples: 'search: quantum computing', 'task: Buy milk', 'list tasks'")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt, OSError):
            print(); break
        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            break
        EMOTION.update_from_text(line, source="user")
        reply = process_message_sync(line, EMOTION.mood_system_prompt())
        EMOTION.update_from_text(reply, source="navi")
        print(f"Navi: {reply}")
        if reply and not reply.startswith("["):
            rate, vol = EMOTION.tts_params(); speak_async(reply[:400], rate=rate, volume=vol)


if __name__ == "__main__":
    if os.environ.get("NETNAVI_RUN_TESTS") == "1":
        run_tests(); sys.exit(0)

    init_db()
    if GUI_AVAILABLE:
        app = QApplication(sys.argv)  # type: ignore[name-defined]
        ensure_api_key_qt()
        window = NetNaviApp()  # type: ignore[name-defined]
        window.show()
        sys.exit(qt_exec(app))
    else:
        # Headless CLI fallback when Qt bindings are absent
        cli_main()
    else:
        # Headless CLI fallback when PyQt5 is absent
        cli_main()
