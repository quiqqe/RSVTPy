#!/usr/bin/env python3
"""
RSVTPy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading, time, json, os, sys
from queue import Queue, Empty
import xml.etree.ElementTree as ET
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import webbrowser
from tkinter import font as tkfont

# Optional HTTP
try:
    import requests

    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "RSVT_config.json")
SAVE_EVERY = 5

# ----------------- Localization -----------------
TRANSLATIONS = {
    "en": {
        "title": "RSVTPy",
        "recent_books": "Recent books / sources:",
        "add_book": "Add Book (txt/fb2)",
        "dynamic_file": "Dynamic File",
        "dynamic_url": "Dynamic URL",
        "dynamic_api": "Dynamic API",
        "words_per_block": "Words/block:",
        "wpm": "WPM:",
        "loop": "Loop",
        "resume": "Resume from last",
        "opacity": "Opacity:",
        "window_size": "Window size:",
        "window_position": "Position:",
        "language": "Language:",
        "launch": "Launch",
        "remove": "Remove",
        "set_start": "Set Start",
        "quit": "Quit",
        "hints": "Right = next, 'a' = auto/manual, Space = pause/play in auto, hold in manual",
        "select_position": "Select start position:",
        "start_from_beginning": "Start from beginning",
        "start_from_last": "Start from last position",
        "start_from_custom": "Custom position...",
        "position_center": "Center",
        "position_top": "Top",
        "position_bottom": "Bottom",
        "api_info": "API running on http://localhost:7141 - Send POST requests with text",
        "settings": "Settings",
        "please_select": "Please select a book or source",
        "no_books": "No recent books. Add a book using the buttons above.",
        "speed": "Speed",
        "display": "Display",
        "behavior": "Behavior",
        "font": "Font:",
        "font_size": "Font size:",
        "warning_eye_movement": "Warning: With these settings, eye movement may be noticeable",
        "auto_window": "Auto window size",
        "custom_window": "Custom size"
    },
    "ru": {
        "title": "RSVTPy",
        "recent_books": "Недавние книги / источники:",
        "add_book": "Добавить книгу (txt/fb2)",
        "dynamic_file": "Динамический файл",
        "dynamic_url": "Динамический URL",
        "dynamic_api": "Динамический API",
        "words_per_block": "Слов/блок:",
        "wpm": "Слов/мин:",
        "loop": "Повтор",
        "resume": "Продолжить с места",
        "opacity": "Прозрачность:",
        "window_size": "Размер окна:",
        "window_position": "Позиция:",
        "language": "Язык:",
        "launch": "Запустить",
        "remove": "Удалить",
        "set_start": "Начало чтения",
        "quit": "Выход",
        "hints": "Вправо = далее, 'a' = авто/ручной, Пробел = пауза/play в авто, удержание в ручном",
        "select_position": "Выберите начало чтения:",
        "start_from_beginning": "С начала",
        "start_from_last": "С последней позиции",
        "start_from_custom": "Выбрать позицию...",
        "position_center": "Центр",
        "position_top": "Вверху",
        "position_bottom": "Внизу",
        "api_info": "API работает на http://localhost:7141 - Отправляйте POST запросы с текстом",
        "settings": "Настройки",
        "please_select": "Пожалуйста, выберите книгу или источник",
        "no_books": "Нет недавних книг. Добавьте книгу с помощью кнопок выше.",
        "speed": "Скорость",
        "display": "Отображение",
        "behavior": "Поведение",
        "font": "Шрифт:",
        "font_size": "Размер шрифта:",
        "warning_eye_movement": "Внимание: При таких настройках движения глаз могут быть заметны",
        "auto_window": "Авто размер окна",
        "custom_window": "Свой размер"
    }
}


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ----------------- Improved FB2 Parser -----------------
def load_fb2(path):
    try:
        with open(path, 'rb') as f:
            raw_content = f.read()

        # Try different encodings
        encodings = ['utf-8', 'windows-1251', 'cp1251', 'koi8-r']
        content = None
        for encoding in encodings:
            try:
                content = raw_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            return "Unable to decode file content"

        # Remove all namespaces for easier parsing
        content = re.sub(r'xmlns="[^"]+"', '', content)

        # Parse XML
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            # Try to fix common XML issues
            content = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', '&amp;', content)
            try:
                root = ET.fromstring(content)
            except ET.ParseError as e:
                return f"XML parsing error: {str(e)}"

        # Extract text from all relevant tags
        text_parts = []

        # Look for body sections first
        bodies = root.findall('.//body')
        if not bodies:
            # If no body found, try to get any text content
            bodies = [root]

        for body in bodies:
            # Get all paragraphs and text elements
            for elem in body.iter():
                tag = elem.tag.lower() if '}' not in elem.tag else elem.tag.split('}')[-1].lower()

                # Include various text-containing elements
                if tag in ['p', 'title', 'subtitle', 'text', 'section', 'poem', 'cite', 'epigraph']:
                    text = ''.join(elem.itertext()).strip()
                    if text and len(text) > 3:  # Filter very short texts
                        # Clean up the text
                        text = re.sub(r'\s+', ' ', text)
                        text_parts.append(text)

                # Also get direct text content
                if elem.text and elem.text.strip():
                    direct_text = elem.text.strip()
                    if len(direct_text) > 3:
                        direct_text = re.sub(r'\s+', ' ', direct_text)
                        text_parts.append(direct_text)

        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in text_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)

        result = "\n\n".join(unique_parts) if unique_parts else "No readable content found in FB2 file"

        # Validate that we got meaningful content
        if len(result) < 100 and "No readable content" not in result:
            return "Very short content extracted - file may be encrypted or in unexpected format"

        return result

    except Exception as e:
        return f"Error reading FB2 file: {str(e)}"


def load_txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="cp1251") as f:
                return f.read()
        except Exception:
            return "Error decoding text file"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def load_file_auto(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fb2":
        return load_fb2(path)
    else:
        return load_txt(path)


# ----------------- Intelligent Text Chunking -----------------
def smart_chunk_text(text, words_per_block=1):
    """Split text into meaningful chunks while respecting sentence boundaries and word limits"""
    if not text or "Error reading" in text or "No readable content" in text:
        return ["No content available"] if not text else [text]

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # First, split by paragraphs then by sentences
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    chunks = []
    for paragraph in paragraphs:
        if not paragraph:
            continue

        # Improved sentence splitting that handles abbreviations
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', paragraph)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            words = sentence.split()

            # If sentence fits in the block, use it as is
            if len(words) <= words_per_block:
                chunks.append(sentence)
            else:
                # Split into chunks at natural break points
                current_chunk = []
                current_word_count = 0

                for i, word in enumerate(words):
                    current_chunk.append(word)
                    current_word_count += 1

                    # Check if we should break here
                    should_break = False

                    # Break if we reached the word limit
                    if current_word_count >= words_per_block:
                        # Try to break at punctuation or conjunctions
                        if i + 1 < len(words):
                            next_word = words[i + 1]
                            # Prefer to break after punctuation or before conjunctions
                            if (word.endswith(('.', '!', '?', ';', ':', ',')) or
                                    next_word.lower() in ['and', 'but', 'or', 'so', 'because', 'however', 'therefore']):
                                should_break = True
                            else:
                                # Otherwise break at word limit
                                should_break = True
                        else:
                            should_break = True

                    if should_break and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(chunk_text)
                        current_chunk = []
                        current_word_count = 0

                # Add any remaining words
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)

    # Fallback: simple word splitting if no chunks created
    if not chunks:
        words = text.split()
        chunks = [" ".join(words[i:i + words_per_block]) for i in range(0, len(words), words_per_block)]

    return chunks


# ----------------- Optimal Window Size Calculator -----------------
def calculate_optimal_window_size(screen_width, screen_height, words_per_block, font_size):
    """Calculate optimal window size based on screen dimensions and reading parameters"""
    # Base calculations for comfortable reading
    max_chars_per_line = min(80, words_per_block * 10)  # Approximate characters per line

    # Calculate width based on characters and font size
    char_width = font_size * 0.6  # Approximate width per character
    optimal_width = min(int(max_chars_per_line * char_width), screen_width * 0.7)

    # Calculate height based on expected lines (with word wrap)
    lines_needed = min(3, max(1, (words_per_block + 4) // 5))  # More words = more lines
    line_height = font_size * 1.5  # Line height including spacing
    optimal_height = min(int(lines_needed * line_height * 1.5), screen_height * 0.3)

    # Ensure minimum sizes
    optimal_width = max(300, optimal_width)
    optimal_height = max(100, optimal_height)

    return (int(optimal_width), int(optimal_height))


# ----------------- API Server -----------------
class TextAPIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            # Try to parse as JSON first
            try:
                data = json.loads(post_data.decode('utf-8'))
                text = data.get('text', '')
            except json.JSONDecodeError:
                # Fallback to form data
                data = urllib.parse.parse_qs(post_data.decode('utf-8'))
                text = data.get('text', [''])[0]

            if text:
                self.server.text_queue.put(text)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps({"status": "success", "message": "Text received"})
                self.wfile.write(response.encode('utf-8'))
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"status": "error", "message": "No text provided"}')

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'{{"status": "error", "message": "Server error: {str(e)}"}}'.encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Disable default logging


class TextAPIServer(threading.Thread):
    def __init__(self, text_queue, port=7141):
        super().__init__(daemon=True)
        self.text_queue = text_queue
        self.port = port
        self.server = None

    def run(self):
        self.server = HTTPServer(('localhost', self.port), TextAPIHandler)
        self.server.text_queue = self.text_queue
        try:
            self.server.serve_forever()
        except Exception:
            pass

    def stop(self):
        if self.server:
            self.server.shutdown()


# ----------------- Dynamic Readers -----------------
class DynamicFileWatcher(threading.Thread):
    def __init__(self, path, out_queue, words_per_block):
        super().__init__(daemon=True)
        self.path = path
        self.q = out_queue
        self.words_per_block = words_per_block
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        seen = ""
        while self.running:
            try:
                if os.path.exists(self.path):
                    with open(self.path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if content != seen:
                        seen = content
                        blocks = smart_chunk_text(content, self.words_per_block)
                        for b in blocks:
                            self.q.put(b)
                time.sleep(1.0)
            except Exception:
                time.sleep(1.0)


class DynamicHTTPPoller(threading.Thread):
    def __init__(self, url, out_queue, words_per_block, interval=2.0):
        super().__init__(daemon=True)
        self.url = url
        self.q = out_queue
        self.words_per_block = words_per_block
        self.interval = interval
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        seen = ""
        while self.running:
            try:
                r = requests.get(self.url, timeout=5)
                if r.status_code == 200:
                    txt = r.text
                    if txt != seen:
                        seen = txt
                        blocks = smart_chunk_text(txt, self.words_per_block)
                        for b in blocks:
                            self.q.put(b)
                time.sleep(self.interval)
            except Exception:
                time.sleep(self.interval)


# ----------------- Reader Window -----------------
class ReaderWindow:
    def __init__(self, master, source_type, source_path, initial_index, words_per_block, wpm, opacity, window_size,
                 loop, config, window_position="center", language="en", font_family="Arial", font_size=16,
                 auto_window_size=True):
        self.master = master
        self.source_type = source_type
        self.source_path = source_path
        self.words_per_block = max(1, int(words_per_block))
        self.wpm = max(10, int(wpm))
        self.opacity = float(opacity)
        self.window_size = window_size
        self.loop = bool(loop)
        self.config = config
        self.window_position = window_position
        self.language = language
        self.font_family = font_family
        self.font_size = font_size
        self.auto_window_size = auto_window_size

        self.static_blocks = []
        self.dynamic_q = Queue()
        self.dynamic_worker = None
        self.api_server = None

        self.index = int(initial_index or 0)
        self.show_count = 0

        self.mode_auto = False
        self.paused = True
        self.running = True
        self.space_held = False
        self.auto_loop_id = None
        self.space_hold_interval = None

        # Show hints before creating the window
        self._show_hints()

        self._build_ui()
        self._load_source()
        self._update_status()
        self.root.focus_force()

    def _show_hints(self):
        hints = {
            "en": "Hints: Right = next, 'a' = toggle auto/manual, Space = pause/play in auto, hold for continuous in manual",
            "ru": "Подсказки: Вправо = далее, 'a' = авто/ручной, Пробел = пауза/play в авто, удержание в ручном"
        }
        messagebox.showinfo(
            TRANSLATIONS[self.language]["title"],
            hints.get(self.language, hints["en"])
        )

    def _build_ui(self):
        self.root = tk.Toplevel(self.master)
        self.root.title(TRANSLATIONS[self.language]["title"])

        # Calculate optimal window size if auto size is enabled
        if self.auto_window_size:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            self.window_size = calculate_optimal_window_size(
                screen_width, screen_height, self.words_per_block, self.font_size
            )

        w, h = self.window_size
        self._position_window(self.root, w, h)
        self.root.attributes("-topmost", True)

        try:
            self.root.attributes("-alpha", self.opacity)
        except Exception:
            pass

        self.root.configure(bg="black")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Modern styling with selected font
        self.style = ttk.Style()

        # Create a frame that will adjust to content
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        # Create label with word wrapping and selected font
        self.label = tk.Label(
            self.main_frame,
            text="",
            font=(self.font_family, self.font_size),
            fg="white",
            bg="black",
            justify="center",
            wraplength=w - 40,  # Enable word wrapping
            anchor="center"
        )
        self.label.grid(row=0, column=0, sticky="nsew")

        self.status = ttk.Label(self.root, text="", font=("Arial", 10), background="black", foreground="lightgray")
        self.status.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Make window resizable and set minsize
        self.root.minsize(300, 100)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Bindings
        self.root.bind("<Right>", self.manual_next)
        self.root.bind("<Left>", self.manual_prev)
        self.root.bind("<KeyPress-space>", self.space_pressed)
        self.root.bind("<KeyRelease-space>", self.space_released)
        self.root.bind("a", self.toggle_mode)
        self.root.bind("q", self.quit)
        self.root.bind("<Escape>", self.quit)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        # Update wrap length when window is resized
        self.root.bind("<Configure>", self._on_window_resize)

    def _on_window_resize(self, event):
        if event.widget == self.root:
            # Update wrap length to match new window width
            new_width = self.root.winfo_width() - 40
            self.label.configure(wraplength=new_width)

    def _position_window(self, win, w, h):
        ws = win.winfo_screenwidth()
        hs = win.winfo_screenheight()

        if self.window_position == "top":
            x = (ws - w) // 2
            y = int(hs * 0.05)
        elif self.window_position == "bottom":
            x = (ws - w) // 2
            y = int(hs * 0.95 - h)
        else:  # center
            x = (ws - w) // 2
            y = (hs - h) // 2

        win.geometry(f"{w}x{h}+{x}+{y}")

    def _load_source(self):
        if self.source_type == "static":
            try:
                txt = load_file_auto(self.source_path)
                if not txt or "No readable content" in txt or "Error reading" in txt:
                    messagebox.showerror("Error", f"Could not read content from file: {txt}")
                    self.quit()
                    return

                self.static_blocks = smart_chunk_text(txt, self.words_per_block)
                if not self.static_blocks:
                    messagebox.showerror("Error", "No readable content found in file")
                    self.quit()
                    return

                if self.index >= len(self.static_blocks):
                    self.index = 0
                self._show_block(self._current_block())

            except Exception as e:
                messagebox.showerror("Load error", f"Can't load file: {e}")
                self.quit()
                return

        elif self.source_type == "dynamic_file":
            self.dynamic_worker = DynamicFileWatcher(self.source_path, self.dynamic_q, self.words_per_block)
            self.dynamic_worker.start()
            self._show_block("Waiting for file updates...")

        elif self.source_type == "dynamic_http":
            if not HAS_REQUESTS:
                messagebox.showerror("Error", "requests library required for HTTP sources")
                self.quit()
                return
            self.dynamic_worker = DynamicHTTPPoller(self.source_path, self.dynamic_q, self.words_per_block)
            self.dynamic_worker.start()
            self._show_block("Waiting for HTTP content...")

        elif self.source_type == "dynamic_api":
            self.api_server = TextAPIServer(self.dynamic_q)
            self.api_server.start()
            self._show_block("API ready. Send text to http://localhost:7141")

        # Calculate proper interval based on WPM
        self.interval = int((60.0 / self.wpm) * self.words_per_block * 1000)
        # For space hold, use the same interval
        self.space_hold_interval = self.interval

    def _current_block(self):
        if 0 <= self.index < len(self.static_blocks):
            return self.static_blocks[self.index]
        return ""

    def _show_block(self, text):
        self.label.config(text=text)
        self._maybe_save_progress()
        self._update_status()

    def _maybe_save_progress(self):
        self.show_count += 1
        if self.show_count % SAVE_EVERY == 0:
            self._save_state()

    def _save_state(self):
        rec = self.config.setdefault("recent_books", {})
        key = self.source_path
        rec.setdefault(key, {})
        rec[key]["last_index"] = self.index
        rec[key]["words_per_block"] = self.words_per_block
        rec[key]["wpm"] = self.wpm
        save_config(self.config)

    def _update_status(self):
        total = len(self.static_blocks)
        mode_text = "AUTO" if self.mode_auto else "MANUAL"
        if self.paused and self.mode_auto:
            mode_text += " [PAUSED]"

        status_text = f"{mode_text} | WPM: {self.wpm} | Block: {self.words_per_block} | Position: {self.index}/{total}"
        if self.source_type.startswith("dynamic"):
            status_text += " | DYNAMIC"
        if self.loop:
            status_text += " | LOOP"

        self.status.config(text=status_text)

    def manual_next(self, event=None):
        if not self._try_dynamic_block():
            if self.index < len(self.static_blocks) - 1:
                self.index += 1
                self._show_block(self._current_block())
            elif self.loop and self.static_blocks:
                self.index = 0
                self._show_block(self._current_block())

    def manual_prev(self, event=None):
        if self.index > 0:
            self.index -= 1
            self._show_block(self._current_block())

    def _try_dynamic_block(self):
        try:
            block = self.dynamic_q.get_nowait()
            self._show_block(block)
            return True
        except Empty:
            return False

    def space_pressed(self, event=None):
        if self.mode_auto:
            self.toggle_pause()
        else:
            self.space_held = True
            self._start_auto_while_held()

    def space_released(self, event=None):
        if not self.mode_auto:
            self.space_held = False
            if self.auto_loop_id:
                self.root.after_cancel(self.auto_loop_id)
                self.auto_loop_id = None

    def _start_auto_while_held(self):
        if not self.space_held:
            return
        self.manual_next()
        if self.space_held:
            # Use the same interval as auto mode for consistent speed
            self.auto_loop_id = self.root.after(self.space_hold_interval, self._start_auto_while_held)

    def toggle_mode(self, event=None):
        self.mode_auto = not self.mode_auto
        if self.mode_auto:
            self.paused = False
            self._start_auto_loop()
        else:
            self.paused = True
            if self.auto_loop_id:
                self.root.after_cancel(self.auto_loop_id)
                self.auto_loop_id = None
        self._update_status()

    def toggle_pause(self, event=None):
        if self.mode_auto:
            self.paused = not self.paused
            if not self.paused:
                self._start_auto_loop()
            self._update_status()

    def _start_auto_loop(self):
        if not self.mode_auto or self.paused:
            return

        if not self._try_dynamic_block():
            if self.index < len(self.static_blocks):
                self._show_block(self._current_block())
                if self.index < len(self.static_blocks) - 1:
                    self.index += 1
                elif self.loop and self.static_blocks:
                    self.index = 0
                else:
                    self.paused = True
                    self._update_status()
                    return

        if self.mode_auto and not self.paused:
            self.root.after(self.interval, self._start_auto_loop)

    def quit(self, event=None):
        self.running = False
        if self.dynamic_worker:
            self.dynamic_worker.stop()
        if self.api_server:
            self.api_server.stop()
        if self.auto_loop_id:
            self.root.after_cancel(self.auto_loop_id)
        self._save_state()
        self.root.destroy()


# ----------------- Improved Launcher -----------------
class Launcher:
    def __init__(self, master):
        self.master = master
        self.config = load_config()
        self.language = self.config.get('language', 'en')
        self.recent_paths = []  # Store paths for listbox items
        self._setup_ui()

    def tr(self, key):
        return TRANSLATIONS[self.language].get(key, key)

    def _setup_ui(self):
        self.master.title(self.tr("title"))
        self.master.geometry("850x650")  # Slightly wider for new options
        self.master.minsize(800, 600)

        # Modern theme
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self._create_widgets()
        self._layout_widgets()
        self._populate_recent()

    def _create_widgets(self):
        # Main frame
        self.main_frame = ttk.Frame(self.master, padding=15)

        # Recent books section - fixed layout
        self.recent_label = ttk.Label(self.main_frame, text=self.tr("recent_books"), font=('Arial', 11, 'bold'))

        # Create a frame for listbox with fixed height to prevent excessive spacing
        self.listbox_frame = ttk.Frame(self.main_frame, height=200)
        self.recent_list = tk.Listbox(self.listbox_frame, font=('Arial', 10))
        self.scrollbar = ttk.Scrollbar(self.listbox_frame, orient="vertical", command=self.recent_list.yview)
        self.recent_list.configure(yscrollcommand=self.scrollbar.set)

        # Button frame
        self.btn_frame = ttk.Frame(self.main_frame)
        self.add_btn = ttk.Button(self.btn_frame, text=self.tr("add_book"), command=self.add_book)
        self.dynamic_file_btn = ttk.Button(self.btn_frame, text=self.tr("dynamic_file"), command=self.add_dynamic_file)
        self.dynamic_url_btn = ttk.Button(self.btn_frame, text=self.tr("dynamic_url"), command=self.add_dynamic_url)
        self.dynamic_api_btn = ttk.Button(self.btn_frame, text=self.tr("dynamic_api"), command=self.add_dynamic_api)

        # Options frame - with proper localization
        self.opt_frame = ttk.LabelFrame(self.main_frame, text=self.tr("settings"), padding=10)

        # Speed settings
        speed_frame = ttk.LabelFrame(self.opt_frame, text=self.tr("speed"), padding=5)
        speed_frame.grid(row=0, column=0, columnspan=2, sticky="we", padx=5, pady=5)

        ttk.Label(speed_frame, text=self.tr("words_per_block")).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.words_spin = ttk.Spinbox(speed_frame, from_=1, to=8, width=8, command=self._check_eye_movement_warning)
        self.words_spin.set("1")
        self.words_spin.grid(row=0, column=1, padx=(0, 15))

        ttk.Label(speed_frame, text=self.tr("wpm")).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.wpm_entry = ttk.Entry(speed_frame, width=8)
        self.wpm_entry.insert(0, "300")
        self.wpm_entry.grid(row=0, column=3)

        # Display settings
        display_frame = ttk.LabelFrame(self.opt_frame, text=self.tr("display"), padding=5)
        display_frame.grid(row=1, column=0, columnspan=2, sticky="we", padx=5, pady=5)

        ttk.Label(display_frame, text=self.tr("font")).grid(row=0, column=0, sticky="w", padx=(0, 5))

        # Font selection
        available_fonts = list(tkfont.families())
        available_fonts.sort()
        # Filter to common readable fonts
        common_fonts = [f for f in available_fonts if
                        any(x in f.lower() for x in ['arial', 'helvetica', 'times', 'courier', 'verdana', 'tahoma'])]
        if not common_fonts:
            common_fonts = available_fonts[:20]  # Limit to first 20 if too many

        self.font_combo = ttk.Combobox(display_frame, values=common_fonts, width=15)
        self.font_combo.set("Arial")
        self.font_combo.grid(row=0, column=1, padx=(0, 15))

        ttk.Label(display_frame, text=self.tr("font_size")).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.font_size_combo = ttk.Combobox(display_frame, values=["12", "14", "16", "18", "20", "24", "28", "32"],
                                            width=5)
        self.font_size_combo.set("16")
        self.font_size_combo.grid(row=0, column=3, padx=(0, 15))

        ttk.Label(display_frame, text=self.tr("opacity")).grid(row=0, column=4, sticky="w", padx=(0, 5))
        self.opacity_combo = ttk.Combobox(display_frame, values=["0.8", "0.85", "0.9", "0.95", "1.0"], width=5)
        self.opacity_combo.set("0.9")
        self.opacity_combo.grid(row=0, column=5, padx=(0, 15))

        # Window settings
        window_frame = ttk.LabelFrame(self.opt_frame, text=self.tr("window_size"), padding=5)
        window_frame.grid(row=2, column=0, columnspan=2, sticky="we", padx=5, pady=5)

        self.window_size_var = tk.StringVar(value="auto")
        self.auto_size_rb = ttk.Radiobutton(window_frame, text=self.tr("auto_window"), variable=self.window_size_var,
                                            value="auto")
        self.auto_size_rb.grid(row=0, column=0, sticky="w", padx=(0, 15))

        self.custom_size_rb = ttk.Radiobutton(window_frame, text=self.tr("custom_window"),
                                              variable=self.window_size_var, value="custom")
        self.custom_size_rb.grid(row=0, column=1, sticky="w", padx=(0, 15))

        ttk.Label(window_frame, text=self.tr("window_position")).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.position_combo = ttk.Combobox(window_frame,
                                           values=[self.tr("position_center"), self.tr("position_top"),
                                                   self.tr("position_bottom")],
                                           width=8)
        self.position_combo.set(self.tr("position_center"))
        self.position_combo.grid(row=0, column=3)

        # Behavior settings
        behavior_frame = ttk.LabelFrame(self.opt_frame, text=self.tr("behavior"), padding=5)
        behavior_frame.grid(row=3, column=0, columnspan=2, sticky="we", padx=5, pady=5)

        self.loop_var = tk.BooleanVar()
        self.loop_cb = ttk.Checkbutton(behavior_frame, text=self.tr("loop"), variable=self.loop_var)
        self.loop_cb.grid(row=0, column=0, padx=(0, 15))

        ttk.Label(behavior_frame, text=self.tr("language")).grid(row=0, column=1, sticky="w", padx=(0, 5))
        self.lang_combo = ttk.Combobox(behavior_frame, values=["English", "Русский"], width=7)
        self.lang_combo.set("English" if self.language == "en" else "Русский")
        self.lang_combo.bind('<<ComboboxSelected>>', self.on_language_change)
        self.lang_combo.grid(row=0, column=2)

        # Warning label
        self.warning_label = ttk.Label(self.opt_frame, text="", foreground="red", font=('Arial', 9, 'bold'))
        self.warning_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=(5, 0))

        # Control buttons
        self.ctrl_frame = ttk.Frame(self.main_frame)
        self.launch_btn = ttk.Button(self.ctrl_frame, text=self.tr("launch"), command=self.launch_selected)
        self.remove_btn = ttk.Button(self.ctrl_frame, text=self.tr("remove"), command=self.remove_selected)
        self.set_start_btn = ttk.Button(self.ctrl_frame, text=self.tr("set_start"), command=self.set_start_position)
        self.quit_btn = ttk.Button(self.ctrl_frame, text=self.tr("quit"), command=self.master.quit)

        # Configure accent button style
        try:
            self.style.configure("Accent.TButton", foreground='white', background='#0078D7')
            self.launch_btn.configure(style="Accent.TButton")
        except Exception:
            # Fallback if style configuration fails
            pass

    def _layout_widgets(self):
        self.main_frame.pack(fill="both", expand=True)

        # Recent books section - fixed layout without excessive spacing
        self.recent_label.pack(anchor="w", pady=(0, 5))

        # Show message if no books
        if not self.config.get("recent_books"):
            no_books_label = ttk.Label(self.main_frame, text=self.tr("no_books"), foreground="gray")
            no_books_label.pack(fill="x", pady=5)

        # Listbox with fixed height
        self.listbox_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.listbox_frame.pack_propagate(False)  # Prevent frame from shrinking
        self.recent_list.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Buttons
        self.btn_frame.pack(fill="x", pady=(0, 15))
        self.add_btn.pack(side="left", padx=(0, 5))
        self.dynamic_file_btn.pack(side="left", padx=5)
        self.dynamic_url_btn.pack(side="left", padx=5)
        self.dynamic_api_btn.pack(side="left", padx=5)

        # Options
        self.opt_frame.pack(fill="x", pady=(0, 15))

        # Control buttons
        self.ctrl_frame.pack(fill="x")
        self.launch_btn.pack(side="left", padx=(0, 5))
        self.remove_btn.pack(side="left", padx=5)
        self.set_start_btn.pack(side="left", padx=5)
        self.quit_btn.pack(side="right")

    def _check_eye_movement_warning(self):
        """Check if current settings might cause noticeable eye movement"""
        try:
            words_per_block = int(self.words_spin.get())
            font_size = int(self.font_size_combo.get())

            # Show warning if too many words or too large font
            if words_per_block > 5 or font_size > 20:
                self.warning_label.config(text=self.tr("warning_eye_movement"))
            else:
                self.warning_label.config(text="")
        except:
            self.warning_label.config(text="")

    def on_language_change(self, event=None):
        lang_map = {"English": "en", "Русский": "ru"}
        self.language = lang_map[self.lang_combo.get()]
        self.config['language'] = self.language
        save_config(self.config)

        # Update UI texts
        self._update_ui_texts()

    def _update_ui_texts(self):
        self.master.title(self.tr("title"))
        self.recent_label.config(text=self.tr("recent_books"))
        self.add_btn.config(text=self.tr("add_book"))
        self.dynamic_file_btn.config(text=self.tr("dynamic_file"))
        self.dynamic_url_btn.config(text=self.tr("dynamic_url"))
        self.dynamic_api_btn.config(text=self.tr("dynamic_api"))
        self.launch_btn.config(text=self.tr("launch"))
        self.remove_btn.config(text=self.tr("remove"))
        self.set_start_btn.config(text=self.tr("set_start"))
        self.quit_btn.config(text=self.tr("quit"))
        self.opt_frame.config(text=self.tr("settings"))

        # Update combobox values
        self.position_combo.config(
            values=[self.tr("position_center"), self.tr("position_top"), self.tr("position_bottom")])
        self.position_combo.set(self.tr("position_center"))

        # Update radio buttons
        self.auto_size_rb.config(text=self.tr("auto_window"))
        self.custom_size_rb.config(text=self.tr("custom_window"))

    def _populate_recent(self):
        self.recent_list.delete(0, "end")
        self.recent_paths = []
        rec = self.config.get("recent_books", {})
        for path, info in rec.items():
            name = os.path.basename(path)
            if info.get('dynamic'):
                name += " [Dynamic]"
            self.recent_list.insert("end", name)
            self.recent_paths.append(path)

    def _get_selected_path(self):
        selection = self.recent_list.curselection()
        if not selection:
            return None
        index = selection[0]
        return self.recent_paths[index] if index < len(self.recent_paths) else None

    def add_book(self):
        path = filedialog.askopenfilename(
            title="Select book file",
            filetypes=[("Books", "*.txt *.fb2"), ("All files", "*.*")],
            initialdir=os.path.expanduser("~/Documents")
        )
        if path:
            self.config.setdefault("recent_books", {})[path] = {"dynamic": False}
            save_config(self.config)
            self._populate_recent()

    def add_dynamic_file(self):
        path = filedialog.askopenfilename(
            title="Select file to watch",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=os.path.expanduser("~/Documents")
        )
        if path:
            self.config.setdefault("recent_books", {})[path] = {"dynamic": True, "type": "file"}
            save_config(self.config)
            self._populate_recent()

    def add_dynamic_url(self):
        if not HAS_REQUESTS:
            messagebox.showerror("Error", "requests library required for HTTP sources")
            return

        url = simpledialog.askstring(self.tr("dynamic_url"), "Enter URL to monitor:")
        if url:
            key = f"url::{url}"
            self.config.setdefault("recent_books", {})[key] = {
                "dynamic": True,
                "type": "http",
                "url": url
            }
            save_config(self.config)
            self._populate_recent()

    def add_dynamic_api(self):
        key = "api::local"
        self.config.setdefault("recent_books", {})[key] = {
            "dynamic": True,
            "type": "api"
        }
        save_config(self.config)
        self._populate_recent()
        messagebox.showinfo("API Info", self.tr("api_info"))

    def set_start_position(self):
        path = self._get_selected_path()
        if not path:
            messagebox.showinfo("Info", self.tr("please_select"))
            return

        info = self.config["recent_books"][path]
        if info.get('dynamic'):
            messagebox.showinfo("Info", "Cannot set start position for dynamic sources")
            return

        # Simple position selection
        try:
            pos = simpledialog.askinteger(
                self.tr("set_start"),
                self.tr("select_position") + "\n(0 = " + self.tr("start_from_beginning") + ")",
                minvalue=0
            )
            if pos is not None:
                info['last_index'] = pos
                save_config(self.config)
                messagebox.showinfo("Success", f"Start position set to {pos}")
        except Exception as e:
            messagebox.showerror("Error", f"Error setting position: {e}")

    def launch_selected(self):
        path = self._get_selected_path()
        if not path:
            messagebox.showinfo("Info", self.tr("please_select"))
            return

        info = self.config["recent_books"][path]

        # Determine source type
        if info.get('dynamic'):
            if info.get('type') == 'file':
                source_type = "dynamic_file"
                source_path = path
            elif info.get('type') == 'http':
                source_type = "dynamic_http"
                source_path = info.get('url', path.split('::', 1)[-1])
            else:  # api
                source_type = "dynamic_api"
                source_path = "api::local"
        else:
            source_type = "static"
            source_path = path

        # Get settings
        try:
            wpb = int(self.words_spin.get())
        except:
            wpb = 1

        try:
            wpm = int(self.wpm_entry.get())
        except:
            wpm = 300

        opacity = float(self.opacity_combo.get())
        font_family = self.font_combo.get()
        font_size = int(self.font_size_combo.get())

        auto_window_size = self.window_size_var.get() == "auto"

        # Set window size
        if auto_window_size:
            # Will be calculated in ReaderWindow
            window_size = (500, 120)  # Default, will be recalculated
        else:
            size_str = "500x120"  # Default size
            win_w, win_h = map(int, size_str.split('x'))
            window_size = (win_w, win_h)

        position_map = {
            self.tr("position_center"): "center",
            self.tr("position_top"): "top",
            self.tr("position_bottom"): "bottom"
        }
        position = position_map[self.position_combo.get()]

        loop = self.loop_var.get()
        last_index = info.get('last_index', 0)

        ReaderWindow(
            self.master, source_type, source_path, last_index, wpb, wpm,
            opacity, window_size, loop, self.config, position, self.language,
            font_family, font_size, auto_window_size
        )

    def remove_selected(self):
        path = self._get_selected_path()
        if path and messagebox.askyesno(self.tr("remove"), "Remove selected item?"):
            self.config["recent_books"].pop(path, None)
            save_config(self.config)
            self._populate_recent()


def main():
    root = tk.Tk()
    app = Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()