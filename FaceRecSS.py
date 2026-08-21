#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import tkinter as tk
import cv2
import numpy as np
import threading
import signal
import ctypes
from datetime import datetime

# Load X11 library for low-level input grabbing (escape-proof)
try:
    x11 = ctypes.CDLL('libX11.so.6')
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    
    x11.XGrabKeyboard.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
    x11.XGrabKeyboard.restype = ctypes.c_int
    
    x11.XGrabPointer.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    x11.XGrabPointer.restype = ctypes.c_int
    
    x11.XSetInputFocus.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    x11.XSetInputFocus.restype = ctypes.c_int
    
    x11.XUngrabKeyboard.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    x11.XUngrabKeyboard.restype = ctypes.c_int
    
    x11.XUngrabPointer.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    x11.XUngrabPointer.restype = ctypes.c_int
    
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.restype = ctypes.c_int
    
    x11.XPending.argtypes = [ctypes.c_void_p]
    x11.XPending.restype = ctypes.c_int
    
    x11.XNextEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    x11.XNextEvent.restype = ctypes.c_int
    
    x11.XLookupString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
    x11.XLookupString.restype = ctypes.c_int
    
    # Custom X11 error handler to ignore protocol errors (Alt-Tab/Focus conflicts)
    XErrorHandlerPrototype = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    
    @XErrorHandlerPrototype
    def x_error_handler(display, error_event):
        return 0
        
    x11.XSetErrorHandler.argtypes = [XErrorHandlerPrototype]
    x11.XSetErrorHandler.restype = XErrorHandlerPrototype
    x11.XSetErrorHandler(x_error_handler)
    
    # Keep reference to prevent garbage collection
    _error_handler_ref = x_error_handler
except Exception as e:
    print(f"Aviso: Não foi possível carregar ou configurar libX11.so.6. Grabs X11 desativados. Erro: {e}", file=sys.stderr)
    x11 = None

# Defaults
DEFAULT_CONFIG = {
    "unlock_sequence": "sai",
    "motion_threshold": 800,
    "camera_index": 0,
    "greeting_enabled": False,
    "fade_duration_ms": 1000,
    "motion_hold_seconds": 5,
    "font_family": "Helvetica",
    "force_logout_minutes": 66,
    "quotes": [
        "Às vezes, tudo que precisamos é de uma frase certa, no momento certo.",
        "May you live every day of your life.",
        "Any fool can know. The point is to understand. - Einstein",
        "“The secret of life, though, is to fall seven times and to get up eight times.” - Paulo Coelho"
    ]
}

def interpolate_color(color1_hex, color2_hex, factor):
    """Interpolate between two hex colors based on a factor (0.0 to 1.0)"""
    factor = max(0.0, min(1.0, factor))
    c1 = [int(color1_hex[i:i+2], 16) for i in (1, 3, 5)]
    c2 = [int(color2_hex[i:i+2], 16) for i in (1, 3, 5)]
    res = [int(c1[i] + (c2[i] - c1[i]) * factor) for i in range(3)]
    return f"#{res[0]:02x}{res[1]:02x}{res[2]:02x}"

class FaceRecognizer:
    """OpenCV 5 face detection (YuNet) + recognition (SFace) via ONNX."""

    def __init__(self, models_dir, faces_dir, confidence_threshold=0.6,
                 recognition_threshold=0.363):
        self.confidence_threshold = confidence_threshold
        self.recognition_threshold = recognition_threshold
        self.known_names = []
        self.known_embeddings = []

        # Load YuNet face detector
        yunet_path = os.path.join(models_dir,
                                  "face_detection_yunet_2023mar.onnx")
        self.detector = cv2.FaceDetectorYN.create(
            yunet_path, "", (320, 320),
            score_threshold=self.confidence_threshold,
            nms_threshold=0.3,
            top_k=5000
        )

        # Load SFace recognizer
        sface_path = os.path.join(models_dir,
                                  "face_recognition_sface_2021dec.onnx")
        self.recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

        # Carregar rostos conhecidos da pasta faces/
        self._load_known_faces(faces_dir)

    def _load_known_faces(self, faces_dir):
        """Carregar rostos conhecidos da pasta faces/."""
        if not os.path.isdir(faces_dir):
            return

        for fname in sorted(os.listdir(faces_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png'):
                continue

            name = os.path.splitext(fname)[0]
            img = cv2.imread(os.path.join(faces_dir, fname))
            if img is None:
                print(f"Aviso: Não foi possível ler {fname}", file=sys.stderr)
                continue

            # Detetar rosto na imagem de referência
            h, w = img.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(img)

            if faces is None or len(faces) == 0:
                print(f"Aviso: Nenhuma face detetada em {fname}",
                      file=sys.stderr)
                continue

            # Use first (highest confidence) detected face
            face = faces[0]
            aligned = self.recognizer.alignCrop(img, face)
            embedding = self.recognizer.feature(aligned)

            self.known_names.append(name)
            self.known_embeddings.append(embedding)

    def recognize(self, frame):
        """Detect and identify faces in frame.
        Returns: list of (name, score) tuples.
        """
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        results = []
        for face in faces:
            try:
                aligned = self.recognizer.alignCrop(frame, face)
                embedding = self.recognizer.feature(aligned)
            except Exception:
                continue

            # Compare against known faces (cosine similarity)
            best_name = "unknown"
            best_score = -1.0

            for name, known_emb in zip(self.known_names,
                                        self.known_embeddings):
                score = self.recognizer.match(
                    embedding, known_emb,
                    getattr(cv2.FaceRecognizerSF, 'FR_COSINE', cv2.FaceRecognizerSF_FR_COSINE)
                )
                if score > best_score:
                    best_score = score
                    best_name = name

            # Cosine: higher = more similar. Threshold ~0.363
            if best_score >= self.recognition_threshold:
                results.append((best_name, best_score))
            else:
                results.append(("unknown", 0.0))

        return results


class MotionDetectorThread(threading.Thread):
    """Background thread to capture frames and perform motion detection"""
    def __init__(self, camera_index, threshold, on_motion_callback,
                 face_recognizer=None, on_face_callback=None,
                 log_callback=None):
        super().__init__()
        self.camera_index = camera_index
        self.threshold = threshold
        self.on_motion_callback = on_motion_callback
        self.face_recognizer = face_recognizer
        self.on_face_callback = on_face_callback
        self.log_callback = log_callback
        self.last_recognition_time = 0
        self.recognition_interval = 1.0  # max 1 recognition per second
        self.stopped = threading.Event()
        self.cap = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None

    def run(self):
        # Abrir fluxo da câmara
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Aviso: Não foi possível abrir a câmara {self.camera_index}", file=sys.stderr)
            return

        ref_frame = None
        was_motion = False
        while not self.stopped.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # Save latest frame thread-safely
            with self.frame_lock:
                self.latest_frame = frame.copy()

            # Resize to reduce CPU load
            resized = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # Initialize reference frame
            if ref_frame is None:
                ref_frame = gray.copy().astype("float")
                continue

            # Update running average
            cv2.accumulateWeighted(gray, ref_frame, 0.05)
            frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(ref_frame))
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Soft Light Filter: check global pixel variance
            non_zero = cv2.countNonZero(thresh)
            total_pixels = 320 * 240
            is_global_light = (non_zero > 0.75 * total_pixels)

            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = False
            if not is_global_light:
                for c in contours:
                    if cv2.contourArea(c) > self.threshold:
                        motion_detected = True
                        break

            self.on_motion_callback(motion_detected)

            # Face recognition: scan immediately on motion start, or on interval
            now_t = time.time()
            just_started_motion = motion_detected and not was_motion
            was_motion = motion_detected

            if (motion_detected and self.face_recognizer
                    and self.on_face_callback):
                if just_started_motion or (now_t - self.last_recognition_time >= self.recognition_interval):
                    self.last_recognition_time = now_t
                    try:
                        results = self.face_recognizer.recognize(frame)
                        if results:
                            names = [f"{n}({s:.2f})" for n, s in results]
                            if self.log_callback:
                                self.log_callback(
                                    f"FaceRec scan: {len(results)} face(s) "
                                    f"-> {', '.join(names)}")
                            self.on_face_callback(results, trigger_frame=frame)
                        else:
                            if self.log_callback:
                                self.log_callback(
                                    "FaceRec scan: nenhuma face detetada")
                    except Exception as e:
                        if self.log_callback:
                            self.log_callback(
                                f"FaceRec erro: {e}")
                        print(f"Erro reconhecimento facial: {e}",
                              file=sys.stderr)

            # Capture around 20 fps
            time.sleep(0.05)

        self.cap.release()

    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        self.stopped.set()

class MotionScreenSaverApp:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.unlock_sequence = self.config.get("unlock_sequence", "amor").lower()
        self.recent_keys = ""
        
        self.alpha = 0.0  # Current fade state (0.0 = black, 1.0 = fully visible)
        self.target_alpha = 0.0
        self.motion_active = False
        self.last_motion_time = 0
        self.input_activity_time = 0.0
        self.last_log_activity_time = 0.0
        
        # Thread safety
        self.motion_detected = False
        self.motion_lock = threading.Lock()
        
        # Determine script directory for resource loading
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Face recognition setup
        models_dir = os.path.join(script_dir, "models")
        faces_dir = os.path.join(script_dir, "faces")
        self.face_recognizer = None
        self.greeting_enabled = self.config.get("greeting_enabled", False)
        self.greeting_text = ""
        self.greeting_show_time = 0
        self.greeting_duration = 5.0
        self.greetings_map = self.config.get("greetings", {})
        self.default_greeting = self.config.get("default_greeting",
                                                 "Olá {name}!")
        self.recognition_cooldown = self.config.get("recognition_cooldown", 10.0)
        self.last_greet_times = {}  # {name: timestamp}
        self.pending_greeting = None  # thread-safe: set by bg thread, read by main
        self.pending_greeting_lock = threading.Lock()

        if os.path.isdir(models_dir) and os.path.isdir(faces_dir):
            try:
                rec_threshold = self.config.get("recognition_threshold", 0.363)
                self.face_recognizer = FaceRecognizer(
                    models_dir, faces_dir,
                    recognition_threshold=rec_threshold
                )
                n = len(self.face_recognizer.known_names)
                names = ', '.join(self.face_recognizer.known_names) or 'nenhuma'
                print(f"Face recognition: {n} pessoa(s) carregada(s)")
                self.log_event(
                    f"FaceRec inicializado: {n} pessoa(s) [{names}], "
                    f"threshold={rec_threshold}")
            except Exception as e:
                print(f"Aviso: Face recognition indisponível: {e}",
                      file=sys.stderr)
                self.log_event(f"FaceRec ERRO inicialização: {e}")
        else:
            self.log_event("FaceRec desativado: pasta models/ ou faces/ "
                          "não encontrada")

        # Load quotes relative to script directory
        self.quotes = self.config.get("quotes", [])
        quotes_file_path = os.path.join(script_dir, "quotes.txt")
        fallback_quotes_path = os.path.join(os.path.dirname(script_dir), "lock", "quotes.txt")
        
        quotes_path = quotes_file_path if os.path.exists(quotes_file_path) else fallback_quotes_path
        if os.path.exists(quotes_path):
            try:
                with open(quotes_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        self.quotes = lines
            except Exception as e:
                print(f"Erro ao ler quotes.txt: {e}", file=sys.stderr)
        self.current_quote = ""
        
        # Open low-level X11 Display
        self.display = None
        if x11:
            try:
                self.display = x11.XOpenDisplay(None)
            except Exception as e:
                print(f"Erro ao abrir display X11: {e}", file=sys.stderr)
        
        # Configure window
        self.root.title("Motion Screen Saver")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.config(bg="black", cursor="none")
        self.root.overrideredirect(True)  # Secure override redirect
        
        # Prevent escaping and force focus
        self.root.focus_force()
        self.root.update()
        
        # Initial X11 low-level grab
        self.grab_inputs()
        
        # Get screen metrics
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        
        # Setup canvas
        self.canvas = tk.Canvas(self.root, width=self.screen_w, height=self.screen_h, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Typography elements
        self.clock_item = self.canvas.create_text(
            self.screen_w // 2, self.screen_h // 2 - 100,
            text="", fill="black", font=(self.config["font_family"], 72, "bold")
        )
        self.date_item = self.canvas.create_text(
            self.screen_w // 2, self.screen_h // 2 - 20,
            text="", fill="black", font=(self.config["font_family"], 22)
        )
        self.quote_item = self.canvas.create_text(
            self.screen_w // 2, self.screen_h // 2 + 80, anchor="center",
            text="", fill="black", font=(self.config["font_family"], 18, "italic"),
            width=self.screen_w - 200, justify="center"
        )
        self.greeting_item = self.canvas.create_text(
            self.screen_w // 2, self.screen_h // 2 + 160,
            text="", fill="black",
            font=(self.config["font_family"], 28, "bold"),
            width=self.screen_w - 200, justify="center"
        )
        self.msg_item = None
        
        # Start motion detection thread
        self.detector = MotionDetectorThread(
            camera_index=self.config["camera_index"],
            threshold=self.config["motion_threshold"],
            on_motion_callback=self.update_motion_state,
            face_recognizer=self.face_recognizer,
            on_face_callback=self.on_face_recognized,
            log_callback=self.log_event
        )
        self.detector.start()
        
        # Timers
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        # Log startup
        self.log_event("Sessão Iniciada / Ecrã Bloqueado")
        
        # Start main update loop
        self.tick()

    def grab_inputs(self):
        """Enforces low-level X11 grabs for escape-proof locking"""
        if x11 and self.display:
            try:
                window_id = self.root.winfo_id()
                x11.XGrabKeyboard(self.display, window_id, False, 1, 1, 0)
                x11.XGrabPointer(self.display, window_id, False, 68, 1, 1, 0, 0, 0)
                x11.XSetInputFocus(self.display, window_id, 2, 0)
            except Exception as e:
                print(f"Erro ao efetuar grab de inputs: {e}", file=sys.stderr)

    def update_motion_state(self, detected):
        with self.motion_lock:
            self.motion_detected = detected

    def on_face_recognized(self, results, trigger_frame=None):
        """Called from detector thread when faces are identified."""
        now = time.time()
        for name, confidence in results:
            if name == "unknown":
                self.log_event("Face desconhecida detetada")
                continue

            # Known face — check cooldown
            last = self.last_greet_times.get(name, 0)
            if now - last < self.recognition_cooldown:
                continue
            self.last_greet_times[name] = now

            # Build greeting if enabled
            if self.greeting_enabled:
                if name in self.greetings_map:
                    greeting = self.greetings_map[name]
                else:
                    greeting = self.default_greeting.format(name=name)

                self.greeting_text = greeting
                self.greeting_show_time = now
                with self.pending_greeting_lock:
                    self.pending_greeting = greeting

            self.log_event(
                f"Face reconhecida: {name} (confiança: {confidence:.2f}) [Saudação: {'ON' if self.greeting_enabled else 'OFF'}]")
            break  # Greet first recognized person only

    def handle_activity(self, event=None):
        self.input_activity_time = time.time()

    def log_event(self, description):
        """Writes a simple text event log to log.txt"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_path = os.path.join(script_dir, "log.txt")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} - {description}\n")
        except Exception as e:
            print(f"Erro ao escrever log: {e}", file=sys.stderr)

    def log_activity(self, source):
        """Logs activity (throttled to 2 seconds)"""
        now = time.time()
        if now - self.last_log_activity_time < 2.0:
            self.log_event(f"Atividade continuada ({source})")
            return
            
        self.last_log_activity_time = now
        self.log_event(f"Atividade detetada ({source})")

    def shutdown(self):
        # Log unlock event before exit
        self.log_event("Sessão Desbloqueada com palavra-passe")
        
        # Stop camera thread
        self.detector.stop()
        self.detector.join(timeout=1.0)
        
        # Free X11 low-level grabs
        if x11 and self.display:
            try:
                x11.XUngrabKeyboard(self.display, 0)
                x11.XUngrabPointer(self.display, 0)
                x11.XCloseDisplay(self.display)
            except Exception:
                pass
                
        self.root.destroy()
        sys.exit(0)

    def tick(self):
        import random
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now
        
        # Keep enforcing X11 grab to prevent bypasses/shortcuts (Alt-Tab, Super Key, etc.)
        self.grab_inputs()
        
        # Read X11 events from our grabbed connection
        if x11 and self.display:
            while x11.XPending(self.display) > 0:
                event = ctypes.create_string_buffer(256)
                x11.XNextEvent(self.display, event)
                event_type = ctypes.cast(event, ctypes.POINTER(ctypes.c_int)).contents.value
                
                # MotionNotify = 6, ButtonPress = 4, KeyPress = 2
                if event_type in (2, 4, 6):
                    source = "Teclado" if event_type == 2 else "Rato"
                    self.handle_activity()
                    self.log_activity(source)
                    
                if event_type == 2:  # KeyPress
                    buf = ctypes.create_string_buffer(32)
                    keysym = ctypes.c_ulong()
                    count = x11.XLookupString(event, buf, 32, ctypes.byref(keysym), None)
                    for i in range(count):
                        try:
                            val = buf[i]
                            if isinstance(val, bytes):
                                char = val.decode('ascii', errors='ignore')
                            elif isinstance(val, int):
                                char = chr(val)
                            else:
                                char = str(val)
                        except Exception:
                            continue
                            
                        if char.isalpha():
                            self.recent_keys += char.lower()
                            if len(self.recent_keys) > len(self.unlock_sequence):
                                self.recent_keys = self.recent_keys[-len(self.unlock_sequence):]
                                
                            if self.recent_keys == self.unlock_sequence:
                                self.shutdown()
                                return  # Stop processing once shutdown
        
        # Check force logout timeout
        if self.config["force_logout_minutes"] > 0:
            elapsed = now - self.start_time
            if elapsed >= (self.config["force_logout_minutes"] * 60):
                self.log_event("Sessão Expirada -> Logout Forçado")
                try:
                    os.kill(-1, signal.SIGKILL)
                except Exception:
                    pass
                sys.exit(1)
                
        # Read motion state thread-safely
        with self.motion_lock:
            camera_motion = self.motion_detected
            
        input_motion = (now - self.input_activity_time) < self.config["motion_hold_seconds"]
        motion = camera_motion or input_motion
            
        if motion:
            if not self.motion_active:
                self.motion_active = True
                if camera_motion:
                    self.log_activity("Camara")
                if self.quotes:
                    self.current_quote = random.choice(self.quotes)
                    self.canvas.itemconfig(self.quote_item, text=self.current_quote)
            self.last_motion_time = now
            self.target_alpha = 0.0
        else:
            if now - self.last_motion_time >= self.config["motion_hold_seconds"]:
                self.motion_active = False
                self.target_alpha = 0.0

        # Animate alpha transition
        fade_speed = 1.0 / (self.config["fade_duration_ms"] / 1000.0)
        if self.alpha < self.target_alpha:
            self.alpha = min(1.0, self.alpha + fade_speed * dt)
        elif self.alpha > self.target_alpha:
            self.alpha = max(0.0, self.alpha - fade_speed * dt)

        # Update clock & date strings
        now_dt = datetime.now()
        clock_str = now_dt.strftime("%H:%M:%S")
        
        weekdays = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
        months = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
        date_str = f"{weekdays[now_dt.weekday()]}, {now_dt.day} de {months[now_dt.month - 1]}"

        text_color = interpolate_color("#000000", "#ffffff", self.alpha)
        muted_color = interpolate_color("#000000", "#aaaaaa", self.alpha)

        self.canvas.itemconfig(self.clock_item, text=clock_str, fill=text_color)
        self.canvas.itemconfig(self.date_item, text=date_str, fill=muted_color)

        if self.quote_item:
            quote_color = interpolate_color("#000000", "#dddddd", self.alpha)
            self.canvas.itemconfig(self.quote_item, fill=quote_color)

        # Check for pending greeting from face recognition thread
        with self.pending_greeting_lock:
            pending = self.pending_greeting
            self.pending_greeting = None
        if pending:
            self.greeting_text = pending
            self.greeting_show_time = now

        # Render face greeting with timed fade
        if self.greeting_text:
            elapsed_greet = now - self.greeting_show_time
            if elapsed_greet < self.greeting_duration:
                if elapsed_greet < 0.5:
                    greet_alpha = elapsed_greet / 0.5
                elif elapsed_greet > self.greeting_duration - 1.0:
                    greet_alpha = (self.greeting_duration - elapsed_greet) / 1.0
                else:
                    greet_alpha = 1.0
                greet_color = interpolate_color("#000000", "#00ff88",
                                                greet_alpha)
                self.canvas.itemconfig(self.greeting_item,
                                       text=self.greeting_text,
                                       fill=greet_color)
            else:
                self.greeting_text = ""
                self.canvas.itemconfig(self.greeting_item, text="",
                                       fill="black")

        self.root.after(16, self.tick)

def main():
    config = DEFAULT_CONFIG.copy()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    fallback_config_path = os.path.join(os.path.dirname(script_dir), "lock", "config.json")
    
    path_to_use = config_path if os.path.exists(config_path) else fallback_config_path
    if os.path.exists(path_to_use):
        try:
            with open(path_to_use, "r") as f:
                user_cfg = json.load(f)
                config.update(user_cfg)
        except Exception as e:
            print(f"Erro ao ler config.json: {e}", file=sys.stderr)

    if sys.stdin.isatty():
        print("=== Configuração do Motion Screen Saver ===")
        try:
            current_greet = "on" if config.get("greeting_enabled", False) else "off"
            greet_val = input(f"Modo de saudação no ecrã (on/off) [{current_greet}]: ").strip().lower()
            if greet_val in ("off", "n", "nao", "não", "0", "false"):
                config["greeting_enabled"] = False
            elif greet_val in ("on", "s", "sim", "1", "true"):
                config["greeting_enabled"] = True

            val = input(f"Tempo até bloquear (minutos ou h:mm) [{config['force_logout_minutes']}]: ").strip()
            if val:
                try:
                    if ':' in val:
                        parts = val.split(':')
                        config["force_logout_minutes"] = int(parts[0]) * 60 + int(parts[1])
                    else:
                        config["force_logout_minutes"] = int(float(val))
                except (ValueError, IndexError):
                    print(f"Aviso: valor inválido '{val}', a usar {config['force_logout_minutes']} minutos.", file=sys.stderr)
                
            import getpass
            pwd = getpass.getpass("Nova palavra-passe de desbloqueio (Enter para manter): ").strip()
            if pwd:
                config["unlock_sequence"] = pwd
        except Exception:
            pass
            
    root = tk.Tk()
    app = MotionScreenSaverApp(root, config)
    
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        signal.signal(signal.SIGTSTP, signal.SIG_IGN)
    except Exception:
        pass

    def handle_sig(signum, frame):
        app.shutdown()

    signal.signal(signal.SIGTERM, handle_sig)
    
    root.mainloop()

if __name__ == "__main__":
    main()
