#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 FAIRY (YOLOv8) + YOUKAI  —  5-панельная система детекции людей и аномалий
Version 7  (полная переработка)

═══════════════════════════════════════════════════════════════════════════
  ПАНЕЛИ:
    Panel 0  — Исходное видео (без изменений)
    Panel 1  — Fairy = YOLOv8: закрашивает КАЖДОГО человека СПЛОШНЫМ ЗЕЛЁНЫМ
    Panel 2  — Fairy-Distort: каждые 5 сек меняет «событие»:
                  BLACKOUT  — замазывает людей чёрным (они «исчезают»)
                  DUPLICATE — дублирует людей (копирует в случайное место)
    Panel 3  — Youkai: обучается на разметке Panel 1, закрашивает людей
               ФИОЛЕТОВЫМ; каждую секунду обновляется
    Panel 4  — Anomaly Engine: Youkai анализирует Panel 2 и ищет аномалии:
                  СКРЫТ!  — человек исчез (BLACKOUT)
                  ДУБЛЬ!  — лишний человек (DUPLICATE)
               Обведено КРАСНЫМ прямоугольником
═══════════════════════════════════════════════════════════════════════════

УСТАНОВКА:
    pip install torch torchvision ultralytics opencv-python pillow matplotlib

ЗАПУСК:
    python FairyYoukai_7v.py
"""

# ── Стандартная библиотека ──────────────────────────────────────────────
import os
import gc
import math
import random
import subprocess
import threading
import time
import traceback
import queue
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Числа и изображения ────────────────────────────────────────────────
import cv2
import numpy as np
from PIL import Image, ImageTk

# ── PyTorch ────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── Matplotlib ─────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── Tkinter ────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog, messagebox

# ── YOLO ───────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _UltralyticsYOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 64)
print("🎯  FAIRY (YOLOv8) + YOUKAI  —  5 панелей")
print(f"    Устройство PyTorch : {DEVICE}")
print(f"    PyTorch            : {torch.__version__}")
print(f"    YOLO (ultralytics) : {'✅ ДОСТУПЕН' if YOLO_AVAILABLE else '❌ pip install ultralytics'}")
print("=" * 64)


# ════════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 1 — ГЛОБАЛЬНЫЕ КОНСТАНТЫ И ТИПЫ ДАННЫХ
# ════════════════════════════════════════════════════════════════════════

# Параметры YOLO
YOLO_MODEL        = 'yolov8n.pt'   # yolov8n / yolov8s / yolov8m — меняйте по желанию
YOLO_CONF         = 0.15           # снижен для выделения всех людей
YOLO_IOU          = 0.35

# Параметры Youkai
Y_ANCHORS         = 24             # количество якорей в голове Youkai
Y_IN_W, Y_IN_H    = 320, 240       # входной размер Youkai
Y_CONF_THR        = 0.26           # порог уверенности при инференсе
Y_ACTIVATE_PREC   = 0.42           # precision для активации Panel 4

# Fairy Distorter
FAIRY_EVENT_SEC   = 8.0            # секунд между сменой события (медленнее)

# UI
UI_W, UI_H        = 1800, 1020     # размер окна
PANEL_W, PANEL_H  = 410, 305       # размер каждой видео-панели
LOG_STEPS         = 80             # логировать в консоль каждые N шагов

# Цвета (RGB для numpy/PIL)
C_GREEN   = (40,  230,  70)
C_PURPLE  = (190,  40, 230)
C_RED     = (230,  40,  40)
C_YELLOW  = (240, 210,  20)
C_CYAN    = (20,  220, 200)
C_ORANGE  = (240, 130,  20)
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,     0,   0)


@dataclass
class Box:
    """
    Нормализованный bounding box (координаты в диапазоне 0..1).
    Поддерживает вычисление IoU, пиксельных координат.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    tag:  str   = ''

    # ── Вычисляемые свойства ─────────────────────────────────────────
    @property
    def cx(self) -> float:  return (self.x1 + self.x2) * 0.5
    @property
    def cy(self) -> float:  return (self.y1 + self.y2) * 0.5
    @property
    def w(self)  -> float:  return max(0.0, self.x2 - self.x1)
    @property
    def h(self)  -> float:  return max(0.0, self.y2 - self.y1)
    @property
    def area(self) -> float: return self.w * self.h
    @property
    def cxcywh(self) -> Tuple[float, float, float, float]:
        return self.cx, self.cy, self.w, self.h

    def pixel(self, W: int, H: int) -> Tuple[int, int, int, int]:
        """Возвращает (x1_px, y1_px, x2_px, y2_px) в пикселях."""
        return (int(np.clip(self.x1, 0, 1) * W),
                int(np.clip(self.y1, 0, 1) * H),
                int(np.clip(self.x2, 0, 1) * W),
                int(np.clip(self.y2, 0, 1) * H))

    def iou(self, other: 'Box') -> float:
        """Intersection over Union с другим боксом."""
        ix1 = max(self.x1, other.x1); iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2); iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 1e-7 else 0.0

    def expand(self, dx: float = 0.0, dy: float = 0.0) -> 'Box':
        """Расширяет бокс на dx (ширина) и dy (высота) в нормализованных единицах."""
        return Box(
            x1=max(0.0, self.x1 - dx),
            y1=max(0.0, self.y1 - dy),
            x2=min(1.0, self.x2 + dx),
            y2=min(1.0, self.y2 + dy),
            conf=self.conf, tag=self.tag
        )


@dataclass
class Anomaly:
    """Зафиксированная аномалия: вид (СКРЫТ/ДУБЛЬ), где, когда."""
    box:       Box
    kind:      str           # 'СКРЫТ' или 'ДУБЛЬ'
    confirmed: bool = False  # Youkai нашёл независимо
    ts:        float = field(default_factory=time.time)


@dataclass
class AppState:
    """
    Общее состояние между потоком обучения и потоком отображения.
    Всё защищено единым lock-ом.
    """
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Готовые кадры для 5 панелей
    panel: Dict[int, Optional[np.ndarray]] = field(
        default_factory=lambda: {i: None for i in range(5)}
    )

    # Текущие детекции
    yolo_boxes:    List[Box] = field(default_factory=list)
    youkai_boxes:  List[Box] = field(default_factory=list)
    fairy_anomalies: List[Anomaly] = field(default_factory=list)

    # Fairy текущее событие
    fairy_event: str = 'BLACKOUT'

    # Метрики Youkai
    youkai_step: int   = 0
    youkai_prec: float = 0.0
    youkai_rec:  float = 0.0

    # Счётчики аномалий
    caught_total: int = 0

    # Система баллов Youkai vs Fairy
    youkai_score: int = 0
    fairy_score:  int = 0

    # История для графиков
    h_prec: deque = field(default_factory=lambda: deque(maxlen=600))
    h_loss: deque = field(default_factory=lambda: deque(maxlen=600))
    h_yolo: deque = field(default_factory=lambda: deque(maxlen=600))

    # Лог аномалий (строки)
    anom_log: List[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 2 — УТИЛИТЫ (ВИДЕО, РИСОВАНИЕ, ТЕНЗОРЫ)
# ════════════════════════════════════════════════════════════════════════

def get_video(path: str = "shibuya_crossing.mp4") -> str:
    """
    Пытается скачать видео перекрёстка Сибуя через yt-dlp.
    Если не получается — генерирует синтетическое видео с пешеходами.
    """
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        print(f"✅ Видео найдено: {path}")
        return path

    print("📥 Пробую скачать видео через yt-dlp...")
    url = "https://www.youtube.com/watch?v=4SvwUbDQZmc"
    try:
        cmd = (f'yt-dlp -f "best[height<=480]" -o "{path}" '
               f'--no-playlist "{url}"')
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=120)
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            print("✅ Видео успешно скачано!")
            return path
        print(f"⚠️  yt-dlp не сработал: {r.stderr[:100]}")
    except Exception as e:
        print(f"⚠️  Ошибка: {e}")

    print("🎬 Генерирую синтетическое видео с пешеходами...")
    return _make_synthetic_video()


def _make_synthetic_video(path: str = "synth_pedestrians.mp4") -> str:
    """
    Создаёт реалистичное синтетическое MP4 (900 кадров, 25 fps).
    Шесть пешеходов движутся по «городской улице»: тело + голова + ноги + руки.
    Добавлены: небо с градиентом, здания с окнами, дорога с разметкой.
    """
    if os.path.exists(path) and os.path.getsize(path) > 50_000:
        return path

    W, H, FPS = 640, 480, 25
    FRAMES     = 900
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, float(FPS), (W, H))

    # Описание пешеходов
    peds = [
        {'x': 60.0,  'y': 328, 'vx':  3.1, 'col': (80, 195, 80),   'w': 36, 'h': 80},
        {'x': 185.0, 'y': 338, 'vx': -4.0, 'col': (100, 155, 220), 'w': 34, 'h': 78},
        {'x': 360.0, 'y': 332, 'vx':  2.7, 'col': (220, 165, 80),  'w': 36, 'h': 80},
        {'x': 505.0, 'y': 336, 'vx': -3.4, 'col': (220, 85,  98),  'w': 34, 'h': 82},
        {'x': 295.0, 'y': 325, 'vx':  4.9, 'col': (165, 75, 210),  'w': 36, 'h': 80},
        {'x': 125.0, 'y': 340, 'vx':  2.1, 'col': (55, 185, 185),  'w': 36, 'h': 78},
    ]

    print("  Генерирую кадры...")
    for fn in range(FRAMES):
        # Создаём фон
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Небо (градиент сверху вниз)
        for yi in range(260):
            t = yi / 260
            r = int(95  + t * 60)
            g = int(120 + t * 55)
            b = int(200 + t * 30)
            frame[yi, :] = (r, g, b)

        # Земля (трава)
        frame[260:355] = (55, 100, 50)

        # Дорога
        cv2.rectangle(frame, (0, 355), (W, 450), (70, 70, 70), -1)

        # Тротуар
        cv2.rectangle(frame, (0, 450), (W, H), (125, 115, 105), -1)

        # Здания слева и справа
        for bx1, bx2, shade in [(0, 90, 72), (550, W, 65)]:
            cv2.rectangle(frame, (bx1, 60), (bx2, 260), (shade, shade, shade+8), -1)
            for wy in range(80, 250, 28):
                for wx in range(bx1 + 8, bx2 - 8, 22):
                    lit = random.random() > 0.25
                    wc  = (220, 220, 170) if lit else (20, 20, 20)
                    cv2.rectangle(frame, (wx, wy), (wx+13, wy+17), wc, -1)

        # Разметка дороги
        for xi in range(0, W, 62):
            cv2.rectangle(frame, (xi, 392), (xi + 44, 400), (195, 195, 195), -1)

        # Рисуем пешеходов
        for p in peds:
            p['x'] += p['vx']
            if p['x'] > W + 55:  p['x'] = -55.0
            if p['x'] < -55:     p['x'] =  W + 55.0

            px, py = int(p['x']), int(p['y'])
            pw, ph = p['w'], p['h']
            col    = p['col']

            # Аниматция ходьбы
            phase    = math.sin(fn * 0.28 + p['x'] * 0.05)
            leg_off  = int(phase * 9)
            arm_off  = int(phase * 11)

            # Тело
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), col, -1)
            # Контур тела (немного темнее)
            dark = tuple(max(0, c - 50) for c in col)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), dark, 1)

            # Голова
            hcx = px + pw // 2
            hcy = py - 14
            cv2.circle(frame, (hcx, hcy), 14, col, -1)
            cv2.circle(frame, (hcx, hcy), 14, dark, 1)

            # Глаза (белые точки)
            cv2.circle(frame, (hcx - 4, hcy - 2), 3, (250, 250, 250), -1)
            cv2.circle(frame, (hcx + 4, hcy - 2), 3, (250, 250, 250), -1)

            # Ноги
            cv2.line(frame,
                     (px + pw//2 - 7, py + ph),
                     (px + pw//2 - 7 + leg_off, py + ph + 18),
                     col, 5)
            cv2.line(frame,
                     (px + pw//2 + 7, py + ph),
                     (px + pw//2 + 7 - leg_off, py + ph + 18),
                     col, 5)

            # Руки
            cv2.line(frame,
                     (px,       py + 20),
                     (px - 10 + arm_off, py + 52),
                     col, 4)
            cv2.line(frame,
                     (px + pw,  py + 20),
                     (px + pw + 10 - arm_off, py + 52),
                     col, 4)

        writer.write(frame)
        if fn % 225 == 0:
            print(f"    кадр {fn}/{FRAMES}")

    writer.release()
    print(f"✅ Синтетическое видео готово: {path}")
    return path


def to_tensor(frame_rgb: np.ndarray) -> torch.Tensor:
    """
    Конвертирует RGB numpy кадр → нормализованный тензор [1, 3, H, W].
    Масштабирует до (Y_IN_W, Y_IN_H) перед конвертацией.
    """
    small = cv2.resize(frame_rgb, (Y_IN_W, Y_IN_H),
                       interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(small).float().div_(255.0)   # [H, W, 3]
    return t.permute(2, 0, 1).unsqueeze_(0)            # [1, 3, H, W]


def draw_box(img: np.ndarray,
             x1: int, y1: int, x2: int, y2: int,
             color: tuple,
             label: str = '',
             thickness: int = 2,
             fill: bool = False) -> None:
    """
    Рисует прямоугольник (и опционально полупрозрачную заливку)
    с читаемым текстом сверху.
    """
    H, W = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    if fill:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.56
        thick = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        ty = max(th + 5, y1)
        lx2 = min(W, x1 + tw + 8)
        cv2.rectangle(img, (x1, ty - th - 5), (lx2, ty + 2), color, -1)
        cv2.putText(img, label, (x1 + 4, ty - 2), font, scale, (0, 0, 0), thick)


def banner(img: np.ndarray,
           text: str,
           bg: tuple = (0, 0, 0),
           fg: tuple = (255, 255, 255),
           y: int = 0) -> None:
    """Рисует полосу-баннер вверху (или в позиции y) изображения."""
    W = img.shape[1]
    cv2.rectangle(img, (0, y), (W, y + 34), bg, -1)
    cv2.putText(img, text, (8, y + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, fg, 2)


def nms(boxes: List[Box], iou_thr: float = 0.45) -> List[Box]:
    """
    Non-Maximum Suppression.
    Убирает дублирующиеся детекции, оставляет самые уверенные.
    """
    if not boxes:
        return []
    srt  = sorted(boxes, key=lambda b: b.conf, reverse=True)
    keep = []
    sup  = set()
    for i, b in enumerate(srt):
        if i in sup:
            continue
        keep.append(b)
        for j in range(i + 1, len(srt)):
            if j not in sup and b.iou(srt[j]) > iou_thr:
                sup.add(j)
    return keep


def iou_box_arr(b: Box, arr: np.ndarray) -> float:
    """
    IoU между Box (x1,y1,x2,y2) и numpy-массивом [cx,cy,w,h].
    Используется для matching GT → якорей.
    """
    ax1 = b.x1;           ay1 = b.y1
    ax2 = b.x2;           ay2 = b.y2
    bx1 = arr[0]-arr[2]/2; by1 = arr[1]-arr[3]/2
    bx2 = arr[0]+arr[2]/2; by2 = arr[1]+arr[3]/2
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    ua = (ax2-ax1)*(ay2-ay1)
    ub = arr[2]*arr[3]
    return inter / max(1e-7, ua + ub - inter)


# ════════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 3 — FAIRY = YOLOv8  (Panel 1 — зелёный детектор)
# ════════════════════════════════════════════════════════════════════════

class FairyYOLO:
    """
    Fairy = YOLOv8n (или любая другая версия YOLO из ultralytics).

    Задачи:
      • Детектировать ТОЛЬКО людей (class 0 в COCO-датасете).
      • Закрашивать их СПЛОШНЫМ ЗЕЛЁНЫМ для Panel 1.
      • Отдавать бокс-список как ground truth для обучения Youkai.

    Модель не обучается — работает только в режиме инференса.
    Чем мощнее YOLO_MODEL, тем точнее разметка для Youkai.
    """

    def __init__(self, model_name: str = YOLO_MODEL):
        if not YOLO_AVAILABLE:
            raise RuntimeError(
                "Библиотека ultralytics не найдена.\n"
                "Установите: pip install ultralytics"
            )
        print(f"  🤖 Загружаю {model_name}...")
        self.model = _UltralyticsYOLO(model_name)
        self.name  = model_name
        self.frames_done   = 0
        self.persons_total = 0
        print(f"  ✅ {model_name} готов!")

    # ──────────────────────────────────────────────────────────────────
    def detect(self, frame_bgr: np.ndarray) -> List[Box]:
        """
        Запускает YOLO на BGR-кадре, возвращает нормализованные боксы людей.
        Фильтрует боксы меньше 8×8 пикселей.
        """
        self.frames_done += 1
        H, W  = frame_bgr.shape[:2]

        results = self.model.predict(
            source  = frame_bgr,
            classes = [0],           # только person (class 0 COCO)
            conf    = YOLO_CONF,
            iou     = YOLO_IOU,
            verbose = False
        )

        out: List[Box] = []
        for res in results:
            if res.boxes is None:
                continue
            for bx in res.boxes:
                x1, y1, x2, y2 = bx.xyxy[0].cpu().tolist()
                cf = float(bx.conf[0].cpu())
                if (x2 - x1) < 8 or (y2 - y1) < 8:
                    continue
                out.append(Box(
                    x1=float(np.clip(x1 / W, 0, 1)),
                    y1=float(np.clip(y1 / H, 0, 1)),
                    x2=float(np.clip(x2 / W, 0, 1)),
                    y2=float(np.clip(y2 / H, 0, 1)),
                    conf=cf, tag='YOLO'
                ))

        self.persons_total += len(out)
        return out

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def render_panel1(frame_rgb: np.ndarray, boxes: List[Box]) -> np.ndarray:
        """
        Panel 1: закрашивает каждого человека СПЛОШНЫМ ЗЕЛЁНЫМ.
        Добавляет яркую рамку и процент уверенности.
        Цель — наглядно показать, что Fairy «видит» всех людей.
        """
        H, W = frame_rgb.shape[:2]
        out  = frame_rgb.copy()

        for b in boxes:
            px1, py1, px2, py2 = b.pixel(W, H)
            # Расширяем вверх чтобы захватить голову
            py1 = max(0, py1 - 20)

            bw = max(1, px2 - px1)
            bh = max(1, py2 - py1)

            # ── Сплошная зелёная заливка региона ────────────────────
            out[py1:py2, px1:px2] = C_GREEN

            # ── Добавляем тёмный контур силуэта ─────────────────────
            cv2.rectangle(out, (px1, py1), (px2, py2), (0, 140, 30), 2)
            # Белая внешняя рамка для видимости
            cv2.rectangle(out, (px1-1, py1-1), (px2+1, py2+1),
                          C_WHITE, 1)

            # ── Подпись ──────────────────────────────────────────────
            label = f"✓ {b.conf:.0%}"
            draw_box(out, px1, py1, px2, py2,
                     color=(0, 200, 50), label=label,
                     thickness=0)

        # Баннер
        banner(out,
               f"🟢  FAIRY = YOLOv8  |  Найдено людей: {len(boxes)}",
               bg=(0, 70, 15), fg=(80, 255, 120))
        return out


# ════════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 4 — FAIRY DISTORTER  (Panel 2 — искажения)
# ════════════════════════════════════════════════════════════════════════

class FairyDistorter:
    """
    Fairy создаёт два вида «аномалий» в видео (Panel 2).
    Каждые FAIRY_EVENT_SEC секунд переключается между событиями:

      BLACKOUT  — замазывает людей чёрным прямоугольником
                  + глитч-шум по краям
                  + красная точка-маркер в центре

      DUPLICATE — копирует тело человека в случайное место кадра
                  + синеватый тинт на копии
                  + мерцающая рамка вокруг копии

    Метод apply() возвращает:
      (искажённый_кадр, список_аномалий_Anomaly)
    Список аномалий используется AnomalyEngine как эталон.
    """

    EVENTS = ['BLACKOUT', 'DUPLICATE']

    def __init__(self):
        self.event         = 'BLACKOUT'
        self._last_switch  = time.time()
        self._switch_count = 0

    # ──────────────────────────────────────────────────────────────────
    def tick(self) -> bool:
        """Проверяет и при необходимости переключает событие. True если переключилось."""
        if time.time() - self._last_switch >= FAIRY_EVENT_SEC:
            idx = self.EVENTS.index(self.event)
            self.event = self.EVENTS[(idx + 1) % len(self.EVENTS)]
            self._last_switch  = time.time()
            self._switch_count += 1
            print(f"  🔄 Fairy: переключилась на {self.event} "
                  f"(переключений: {self._switch_count})")
            return True
        return False

    # ──────────────────────────────────────────────────────────────────
    def apply(self,
              frame_rgb: np.ndarray,
              yolo_boxes: List[Box]
              ) -> Tuple[np.ndarray, List[Anomaly]]:
        """
        Применяет текущее событие к кадру.
        Возвращает (distorted_frame, anomaly_list).
        """
        self.tick()
        if self.event == 'BLACKOUT':
            out, anoms = self._blackout(frame_rgb, yolo_boxes)
        else:
            out, anoms = self._duplicate(frame_rgb, yolo_boxes)

        self._draw_panel2_banner(out)
        return out, anoms

    # ──────────────────────────────────────────────────────────────────
    def _blackout(self,
                  frame_rgb: np.ndarray,
                  boxes: List[Box]
                  ) -> Tuple[np.ndarray, List[Anomaly]]:
        """
        Событие BLACKOUT:
          • Закрашивает тело+голову каждого человека чёрным.
          • Добавляет полосатый цифровой шум по краям (реалистичный артефакт цензуры).
          • Ставит красную точку в центре — «маркер цензора».
        """
        H, W  = frame_rgb.shape[:2]
        out   = frame_rgb.copy()
        anoms: List[Anomaly] = []

        for b in boxes:
            # Немного расширяем вверх для головы
            b_exp = b.expand(dy=0.025)
            px1, py1, px2, py2 = b_exp.pixel(W, H)
            bw = max(1, px2 - px1)
            bh = max(1, py2 - py1)

            # Чёрная заливка
            out[py1:py2, px1:px2] = 0

            # Глитч-шум у левого края (эффект цифровой помехи)
            nw = max(3, bw // 5)
            noise_l = np.random.randint(0, 55, (bh, nw, 3), dtype=np.uint8)
            out[py1:py2, px1:min(W, px1+nw)] = noise_l

            # Горизонтальные помехи (scanlines)
            for scan_y in range(py1 + 5, py2 - 5, 12):
                scan_x2 = min(W, px1 + random.randint(bw//3, bw))
                out[scan_y:scan_y+2, px1:scan_x2] = (
                    random.randint(20, 80),
                    random.randint(0, 40),
                    random.randint(0, 40)
                )

            # Красная точка-маркер в центре
            cx_px = (px1 + px2) // 2
            cy_px = (py1 + py2) // 2
            cv2.circle(out, (cx_px, cy_px), 5, (220, 30, 30), -1)
            cv2.circle(out, (cx_px, cy_px), 7, (255, 80, 80), 1)

            anoms.append(Anomaly(
                box  = Box(b_exp.x1, b_exp.y1, b_exp.x2, b_exp.y2, b.conf),
                kind = 'СКРЫТ'
            ))

        return out, anoms

    # ──────────────────────────────────────────────────────────────────
    def _duplicate(self,
                   frame_rgb: np.ndarray,
                   boxes: List[Box]
                   ) -> Tuple[np.ndarray, List[Anomaly]]:
        """
        Событие DUPLICATE:
          • Копирует регион каждого человека в случайное место кадра.
          • Добавляет синеватый тинт на копию (отличие от оригинала).
          • Рисует мерцающую рамку вокруг копии.
        """
        H, W  = frame_rgb.shape[:2]
        out   = frame_rgb.copy()
        anoms: List[Anomaly] = []

        for b in boxes:
            b_exp = b.expand(dy=0.025)
            px1, py1, px2, py2 = b_exp.pixel(W, H)
            bw = max(5, px2 - px1)
            bh = max(5, py2 - py1)

            # Ищем место для копии, избегая оригинала (30 попыток)
            placed = False
            for _ in range(30):
                nx  = random.randint(5, max(6, W - bw - 5))
                ny  = random.randint(5, max(6, H - bh - 5))
                # Проверка: не перекрывает оригинал?
                ox = not (nx + bw < px1 or nx > px2)
                oy = not (ny + bh < py1 or ny > py2)
                if not (ox and oy):
                    placed = True
                    break

            if not placed:
                continue

            nx2 = min(W, nx + bw)
            ny2 = min(H, ny + bh)
            aw  = nx2 - nx   # actual width (может урезаться у края)
            ah  = ny2 - ny

            if aw <= 0 or ah <= 0:
                continue

            # Копируем пиксели
            crop = frame_rgb[py1:py1+ah, px1:px1+aw].copy()

            # Синеватый тинт (артефакт клонирования)
            tint = np.zeros_like(crop, dtype=np.int16)
            tint[:, :, 0] =  25   # +R
            tint[:, :, 2] =  50   # +B
            tint[:, :, 1] = -15   # -G
            crop = np.clip(crop.astype(np.int16) + tint, 0, 255).astype(np.uint8)

            # Лёгкое размытие копии (нереалистичная "призрачность")
            crop = cv2.GaussianBlur(crop, (3, 3), 0)

            out[ny:ny2, nx:nx2] = crop

            # Мерцающая рамка вокруг дубля
            flicker_r = random.randint(180, 255)
            cv2.rectangle(out, (nx, ny), (nx2, ny2),
                          (flicker_r, 30, 30), 1)

            anoms.append(Anomaly(
                box  = Box(nx/W, ny/H, nx2/W, ny2/H, b.conf),
                kind = 'ДУБЛЬ'
            ))

        return out, anoms

    # ──────────────────────────────────────────────────────────────────
    def _draw_panel2_banner(self, img: np.ndarray) -> None:
        """Рисует информационный баннер о текущем событии."""
        if self.event == 'BLACKOUT':
            bg = (40, 10, 10)
            fg = (255, 110, 110)
            txt = "⚡  FAIRY: СКРЫВАЕТ ЛЮДЕЙ  [BLACKOUT]"
        else:
            bg = (20, 10, 40)
            fg = (180, 130, 255)
            txt = "⚡  FAIRY: ДУБЛИРУЕТ ЛЮДЕЙ  [DUPLICATE]"
        banner(img, txt, bg=bg, fg=fg)


# ════════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 5 — YOUKAI (нейросеть детекции людей)
# ════════════════════════════════════════════════════════════════════════

class Youkai(nn.Module):
    """
    Youkai — кастомная нейросеть детекции людей.

    Архитектура:
      Backbone: 4 блока Conv-BN-LeakyReLU + MaxPool
                (глубокий извлечитель признаков от 3-канального входа до 256ch)
      Neck: Multi-Scale Pooling — объединяет признаки из block3 (128ch)
            и block4 (256ch) через AdaptiveAvgPool → Linear слой
      Head: 3-слойный MLP → NUM_ANCHORS * 5 чисел
            5 чисел на якорь: cx, cy, w, h, conf (все в [0..1] после sigmoid)

    Вход:  [B, 3, Y_IN_H, Y_IN_W]
    Выход: boxes [B, A, 4],  conf [B, A, 1]
    """

    def __init__(self, num_anchors: int = Y_ANCHORS):
        super().__init__()
        self.num_anchors = num_anchors
        A = num_anchors

        # ── Backbone блок 1: 3 → 32 ──────────────────────────────────
        self.blk1 = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)   # H/2, W/2
        )

        # ── Backbone блок 2: 32 → 64 ─────────────────────────────────
        self.blk2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)   # H/4, W/4
        )

        # ── Backbone блок 3: 64 → 128 ────────────────────────────────
        self.blk3 = nn.Sequential(
            nn.Conv2d(64,  128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)   # H/8, W/8
        )

        # ── Backbone блок 4: 128 → 256 (с residual shortcut) ─────────
        self.blk4_main = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
        )
        self.blk4_skip = nn.Conv2d(128, 256, 1, bias=False)
        self.blk4_act  = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)   # H/16, W/16
        )

        # ── Neck: multi-scale pooling ─────────────────────────────────
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))   # 128 * 16
        self.pool4 = nn.AdaptiveAvgPool2d((4, 4))   # 256 * 16

        # 128*16 + 256*16 = 2048 + 4096 = 6144
        self.neck = nn.Sequential(
            nn.Linear(6144, 1024, bias=False),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25),
        )

        # ── Head: предсказание якорей ─────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(512,  256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256,  A * 5)
        )

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        # Backbone
        x1 = self.blk1(x)
        x2 = self.blk2(x1)
        x3 = self.blk3(x2)

        # Блок 4 с residual shortcut
        main = self.blk4_main(x3)
        skip = self.blk4_skip(x3)
        x4   = self.blk4_act(main + skip)

        # Neck: multi-scale
        f3   = self.pool3(x3).flatten(1)        # [B, 2048]
        f4   = self.pool4(x4).flatten(1)        # [B, 4096]
        feat = self.neck(torch.cat([f3, f4], 1))  # [B, 1024]

        # Head
        raw  = self.head(feat).view(-1, self.num_anchors, 5)

        boxes = torch.sigmoid(raw[:, :, :4])     # cx, cy, w, h → [0,1]
        conf  = torch.sigmoid(raw[:, :, 4:5])    # confidence  → [0,1]
        return boxes, conf


# ════════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 6 — ТРЕНЕР YOUKAI
# ════════════════════════════════════════════════════════════════════════

class YoukaiTrainer:
    """
    Управляет жизненным циклом Youkai:
      • Обучение (train_step) на GT-боксах от YOLO
      • Инференс (detect) для получения боксов
      • Метрики: precision, recall
      • Рендер Panel 3 (фиолетовая заливка)
      • Сохранение и загрузка чекпоинтов
    """

    def __init__(self):
        print("  🔮 Создаю Youkai (multi-scale backbone)...")
        self.net = Youkai(Y_ANCHORS).to(DEVICE)

        # AdamW — хорошо для задач детекции
        self.opt = optim.AdamW(self.net.parameters(),
                               lr=1e-3, weight_decay=1e-4)

        # Cosine Annealing с перезапусками — помогает выйти из локальных минимумов
        self.sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=400, T_mult=2, eta_min=5e-6
        )

        # Метрики
        self.step          = 0
        self.best_prec     = 0.0
        self.h_loss        = deque(maxlen=600)
        self.h_prec        = deque(maxlen=600)
        self.h_recall      = deque(maxlen=600)
        self.h_yolo_count  = deque(maxlen=600)

        print("  ✅ Youkai инициализирован!")

    # ──────────────────────────────────────────────────────────────────
    def train_step(self,
                   frame_t: torch.Tensor,
                   gt: List[Box]) -> Tuple[float, float, float]:
        """
        Один шаг supervised learning.

          frame_t : [1, 3, H, W]  (нормализованный тензор кадра)
          gt      : список Box от YOLO — ground truth

        Алгоритм matching GT→якорь:
          1. Ищем якорь с максимальным IoU с GT-боксом
          2. Если IoU == 0 для всех (никакого перекрытия) —
             берём якорь ближайший по центроиду
          3. Для позитивных якорей считаем SmoothL1 по координатам
          4. BCE по conf с усиленным весом для позитивов (нет людей → редкость)

        Возвращает (loss, precision, recall).
        """
        frame_t = frame_t.to(DEVICE)
        A = Y_ANCHORS
        n = len(gt)

        self.net.train()
        self.opt.zero_grad(set_to_none=True)

        pred_boxes, pred_conf = self.net(frame_t)
        # pred_boxes: [1, A, 4],  pred_conf: [1, A, 1]

        # ── Таргеты ───────────────────────────────────────────────────
        tgt_b = torch.zeros(1, A, 4, device=DEVICE)
        tgt_c = torch.zeros(1, A, 1, device=DEVICE)
        assigned: set = set()

        pb_np = pred_boxes[0].detach().cpu().numpy()  # [A, 4]

        for b in gt[:A]:
            gt_cx = (b.x1 + b.x2) * 0.5
            gt_cy = (b.y1 + b.y2) * 0.5
            gt_w  = b.x2 - b.x1
            gt_h  = b.y2 - b.y1
            gt_arr = np.array([gt_cx, gt_cy, gt_w, gt_h])

            best_ai = -1
            best_iou = -1.0
            best_dist = float('inf')

            for ai in range(A):
                if ai in assigned:
                    continue
                iou_v = iou_box_arr(b, pb_np[ai])
                dist  = (pb_np[ai, 0] - gt_cx)**2 + (pb_np[ai, 1] - gt_cy)**2
                if iou_v > best_iou:
                    best_iou, best_ai = iou_v, ai
                if dist < best_dist:
                    best_dist = dist
                    if best_iou < 0.01:
                        best_ai = ai

            if best_ai >= 0:
                tgt_b[0, best_ai] = torch.tensor(gt_arr, dtype=torch.float32,
                                                  device=DEVICE)
                tgt_c[0, best_ai, 0] = 1.0
                assigned.add(best_ai)

        # ── Маски позитивных / негативных якорей ─────────────────────
        pos = tgt_c.squeeze(-1) > 0.5   # [1, A] bool
        neg = ~pos

        # ── Регрессионный лосс (только позитивные) ────────────────────
        box_loss = (
            F.smooth_l1_loss(pred_boxes[pos], tgt_b[pos], beta=0.04)
            if pos.any() else torch.zeros(1, device=DEVICE)
        )

        # ── Confidence BCE ────────────────────────────────────────────
        pos_w = 6.5 if n > 0 else 1.0
        cf_pos = (
            F.binary_cross_entropy(pred_conf[pos], tgt_c[pos])
            if pos.any() else torch.zeros(1, device=DEVICE)
        )
        cf_neg = (
            F.binary_cross_entropy(pred_conf[neg], tgt_c[neg])
            if neg.any() else torch.zeros(1, device=DEVICE)
        )

        loss = box_loss * pos_w * 2.2 + cf_pos * pos_w + cf_neg * 0.38

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.5)
        self.opt.step()
        self.sched.step()
        self.step += 1

        # ── Метрики ───────────────────────────────────────────────────
        prec, rec = self._metrics(pred_boxes, pred_conf, gt)

        self.h_loss.append(float(loss))
        self.h_prec.append(prec)
        self.h_recall.append(rec)
        self.h_yolo_count.append(n)

        if prec > self.best_prec:
            self.best_prec = prec

        return float(loss), prec, rec

    # ──────────────────────────────────────────────────────────────────
    def _metrics(self,
                 pred_boxes: torch.Tensor,
                 pred_conf:  torch.Tensor,
                 gt: List[Box],
                 cf_thr: float = 0.22,
                 iou_thr: float = 0.32
                 ) -> Tuple[float, float]:
        """Precision и Recall Youkai относительно GT-боксов YOLO."""
        if not gt:
            # Нет людей → правильно если Youkai тоже ничего не нашёл
            fp = int(torch.sum(pred_conf[0, :, 0] > cf_thr).item())
            return (1.0 if fp == 0 else 0.0), 1.0

        with torch.no_grad():
            pb = pred_boxes[0].cpu().numpy()
            pc = pred_conf[0, :, 0].cpu().numpy()

        preds = [(pb[ai], pc[ai]) for ai in range(Y_ANCHORS)
                 if pc[ai] > cf_thr]

        if not preds:
            return 0.0, 0.0

        TP = 0
        matched = set()
        for p_arr, _ in preds:
            for gi, g in enumerate(gt):
                if gi in matched:
                    continue
                if iou_box_arr(g, p_arr) > iou_thr:
                    TP += 1
                    matched.add(gi)
                    break

        FP = len(preds) - TP
        FN = len(gt)    - TP
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        return prec, rec

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def detect(self, frame_t: torch.Tensor,
               cf_thr: float = Y_CONF_THR) -> List[Box]:
        """
        Инференс Youkai с NMS.
        Возвращает список Box в нормализованных координатах.
        """
        self.net.eval()
        preds, confs = self.net(frame_t.to(DEVICE))
        self.net.train()

        pb = preds[0].cpu().numpy()
        pc = confs[0, :, 0].cpu().numpy()

        raw: List[Box] = []
        for ai in range(Y_ANCHORS):
            if pc[ai] < cf_thr:
                continue
            cx, cy, w, h = pb[ai]
            x1 = float(np.clip(cx - w * 0.5, 0, 1))
            y1 = float(np.clip(cy - h * 0.5, 0, 1))
            x2 = float(np.clip(cx + w * 0.5, 0, 1))
            y2 = float(np.clip(cy + h * 0.5, 0, 1))
            if (x2 - x1) < 0.01 or (y2 - y1) < 0.01:
                continue
            raw.append(Box(x1, y1, x2, y2, conf=float(pc[ai]), tag='YOUKAI'))

        return nms(raw, iou_thr=0.42)

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def render_panel3(frame_rgb: np.ndarray,
                      boxes:     List[Box],
                      step:      int,
                      prec:      float) -> np.ndarray:
        """
        Panel 3: закрашивает людей СПЛОШНЫМ ФИОЛЕТОВЫМ.
        Включает прогресс-бар precision и отметку порога активации Panel 4.
        """
        H, W = frame_rgb.shape[:2]
        out  = frame_rgb.copy()

        for b in boxes:
            px1, py1, px2, py2 = b.pixel(W, H)
            py1 = max(0, py1 - 18)
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(W, px2), min(H, py2)
            if px2 <= px1 or py2 <= py1:
                continue

            # Сплошная фиолетовая заливка
            out[py1:py2, px1:px2] = C_PURPLE

            # Обводка (насыщенность зависит от conf)
            br = int(160 + 95 * b.conf)
            cv2.rectangle(out, (px1, py1), (px2, py2),
                          (br, 20, 230), 3)
            cv2.rectangle(out, (px1-1, py1-1), (px2+1, py2+1),
                          C_WHITE, 1)

            label = f"🔮 {b.conf:.0%}"
            draw_box(out, px1, py1, px2, py2,
                     (170, 20, 210), label=label, thickness=0)

        # Баннер
        banner(out,
               f"🔮  YOUKAI  шаг:{step}  precision:{prec:.1%}",
               bg=(40, 0, 70), fg=(210, 80, 255))

        # Прогресс-бар precision
        bar_x = 10;  bar_y = 36
        bar_w = W - 20; bar_h = 9
        cv2.rectangle(out, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h),
                      (30, 0, 50), -1)
        fill = int(bar_w * min(1.0, prec))
        if fill > 0:
            bar_col = (
                int(200 * prec + 50),
                int(20  * prec),
                int(255 - 180 * prec)
            )
            cv2.rectangle(out, (bar_x, bar_y),
                          (bar_x+fill, bar_y+bar_h), bar_col, -1)

        # Метка порога активации Panel 4
        thr_x = bar_x + int(bar_w * Y_ACTIVATE_PREC)
        cv2.line(out, (thr_x, bar_y-3), (thr_x, bar_y+bar_h+3),
                 C_YELLOW, 2)
        cv2.putText(out, f"P4≥{int(Y_ACTIVATE_PREC*100)}%",
                    (thr_x - 18, bar_y + bar_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_YELLOW, 1)
        return out

    # ──────────────────────────────────────────────────────────────────
    def save(self, path: str = 'youkai_v7.pth') -> bool:
        try:
            torch.save({
                'net':       self.net.state_dict(),
                'opt':       self.opt.state_dict(),
                'sched':     self.sched.state_dict(),
                'step':      self.step,
                'best_prec': self.best_prec,
            }, path)
            print(f"  💾 Сохранено → {path}  "
                  f"(шаг {self.step}, best_prec {self.best_prec:.1%})")
            return True
        except Exception as e:
            print(f"  ⚠️  Ошибка сохранения: {e}")
            return False

    def load(self, path: str = 'youkai_v7.pth') -> bool:
        if not os.path.exists(path):
            return False
        try:
            d = torch.load(path, map_location=DEVICE, weights_only=False)
            self.net.load_state_dict(d['net'])
            self.opt.load_state_dict(d['opt'])
            if 'sched' in d:
                self.sched.load_state_dict(d['sched'])
            self.step      = d.get('step', 0)
            self.best_prec = d.get('best_prec', 0.0)
            print(f"  📂 Загружено ← {path}  "
                  f"(шаг {self.step}, best_prec {self.best_prec:.1%})")
            return True
        except Exception as e:
            print(f"  ⚠️  Ошибка загрузки: {e}")
            return False


# ════════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 7 — ANOMALY ENGINE  (Panel 4)
# ════════════════════════════════════════════════════════════════════════

class AnomalyEngine:
    """
    Анализирует искажённый кадр (Panel 2) и ищет аномалии (Panel 4).

    Логика работы:
    ──────────────────────────────────────────────────────────────────
    Активируется когда precision Youkai ≥ Y_ACTIVATE_PREC.

    Запускает Youkai на искажённом кадре (Panel 2) и сравнивает результат
    с детекциями YOLO на оригинальном кадре.

    Matching через IoU (порог 0.28):
      • Каждый YOLO-бокс пытается найти пару среди Youkai-боксов.
      • Неспаренные YOLO-боксы (при BLACKOUT) — человек был скрыт → СКРЫТ!
      • Неспаренные Youkai-боксы (при DUPLICATE) — лишний клон → ДУБЛЬ!

    Вокруг аномалии рисуется КРАСНАЯ рамка + анимированный flash.
    ──────────────────────────────────────────────────────────────────
    """

    IOU_MATCH = 0.27

    def __init__(self):
        self.caught_total = 0
        self.missed_total = 0
        self.youkai_score = 0
        self.fairy_score  = 0
        self.event_log: List[str]             = []
        self._flashes:  List[Tuple[Box, float]] = []  # (box, expire)
        self._last_analysis_time = time.time()
        self._analysis_delay = 0.35  # замедление для наглядности

    # ──────────────────────────────────────────────────────────────────
    def analyze(self,
                distorted_rgb:  np.ndarray,
                trainer:        YoukaiTrainer,
                yolo_boxes:     List[Box],
                fairy_anoms:    List[Anomaly],
                fairy_event:    str,
                youkai_prec:    float
                ) -> Tuple[np.ndarray, List[Anomaly]]:
        """
        Главный метод. Возвращает (panel4_frame, caught_anomalies).
        """
        out    = distorted_rgb.copy()
        caught: List[Anomaly] = []

        # ── Замедление для наглядности ────────────────────────────────
        now = time.time()
        if now - self._last_analysis_time < self._analysis_delay:
            self._draw_waiting(out, youkai_prec)
            return out, caught
        self._last_analysis_time = now

        # ── Ещё не обучен достаточно ──────────────────────────────────
        if youkai_prec < Y_ACTIVATE_PREC:
            self._draw_waiting(out, youkai_prec)
            return out, caught

        # ── Запускаем Youkai на искажённом кадре ─────────────────────
        ft         = to_tensor(distorted_rgb)
        cf_thr     = max(0.14, Y_CONF_THR - 0.06)   # чуть ниже для чувствительности
        yk_on_dist = trainer.detect(ft, cf_thr=cf_thr)

        H, W = out.shape[:2]

        # ── Matching YOLO ↔ Youkai ────────────────────────────────────
        unm_yolo  = list(range(len(yolo_boxes)))
        unm_youk  = list(range(len(yk_on_dist)))

        matched_yolo = set()
        matched_youk = set()

        for yi, yb in enumerate(yolo_boxes):
            best_yki = -1
            best_iou = self.IOU_MATCH
            for yki, ykb in enumerate(yk_on_dist):
                if yki in matched_youk:
                    continue
                iv = yb.iou(ykb)
                if iv > best_iou:
                    best_iou, best_yki = iv, yki
            if best_yki >= 0:
                matched_yolo.add(yi)
                matched_youk.add(best_yki)

        # ── СКРЫТ: YOLO видел, Youkai в искажённом не нашёл ──────────
        if fairy_event == 'BLACKOUT':
            for yi in range(len(yolo_boxes)):
                if yi in matched_yolo:
                    continue
                b = yolo_boxes[yi]
                px1, py1, px2, py2 = b.pixel(W, H)
                py1 = max(0, py1 - 20)

                # Красная рамка (толстая)
                for th in (7, 3):
                    cv2.rectangle(out, (px1, py1), (px2, py2), C_RED, th)
                cv2.rectangle(out, (px1-1, py1-1), (px2+1, py2+1), C_WHITE, 1)

                # Крест поверх скрытой области
                mx, my = (px1+px2)//2, (py1+py2)//2
                cv2.line(out, (mx-16, my-16), (mx+16, my+16), C_RED, 3)
                cv2.line(out, (mx+16, my-16), (mx-16, my+16), C_RED, 3)

                draw_box(out, px1, py1, px2, py2,
                         C_RED,
                         label=f"⚠ СКРЫТ!  {b.conf:.0%}",
                         thickness=0)

                ae = Anomaly(Box(b.x1, b.y1, b.x2, b.y2, b.conf),
                             kind='СКРЫТ', confirmed=True)
                caught.append(ae)
                self.caught_total += 1
                self.youkai_score += 1
                ts  = time.strftime('%H:%M:%S')
                msg = f"[{ts}] ✅ СКРЫТ НАЙДЕН!  ({mx},{my})  Youkai+1 ★"
                self._log(msg)
                self._flashes.append((ae.box, time.time() + 0.9))

        # ── ДУБЛЬ: Youkai нашёл в искажённом, YOLO не видел ──────────
        elif fairy_event == 'DUPLICATE':
            for yki in range(len(yk_on_dist)):
                if yki in matched_youk:
                    continue
                b = yk_on_dist[yki]
                px1, py1, px2, py2 = b.pixel(W, H)

                for th in (7, 3):
                    cv2.rectangle(out, (px1, py1), (px2, py2), C_RED, th)
                cv2.rectangle(out, (px1-1, py1-1), (px2+1, py2+1), C_WHITE, 1)

                mx, my = (px1+px2)//2, (py1+py2)//2
                cv2.putText(out, "×2", (mx-15, my+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 60, 60), 3)

                draw_box(out, px1, py1, px2, py2,
                         C_RED,
                         label=f"⚠ ДУБЛЬ!  {b.conf:.0%}",
                         thickness=0)

                ae = Anomaly(Box(b.x1, b.y1, b.x2, b.y2, b.conf),
                             kind='ДУБЛЬ', confirmed=True)
                caught.append(ae)
                self.caught_total += 1
                self.youkai_score += 1
                ts  = time.strftime('%H:%M:%S')
                msg = f"[{ts}] ✅ ДУБЛЬ НАЙДЕН!  ({mx},{my})  Youkai+1 ★"
                self._log(msg)
                self._flashes.append((ae.box, time.time() + 0.9))

        # Если были созданы аномалии но не найдены - балл Fairy
        if len(fairy_anoms) > 0 and len(caught) == 0:
            self.fairy_score += 1
            ts  = time.strftime('%H:%M:%S')
            msg = f"[{ts}] ❌ ОШИБКА ПРОПУЩЕНА!  Fairy+1 ★"
            self._log(msg)

        # ── Нормальные (совпавшие) боксы — тихая зелёная рамка ───────
        for yi in matched_yolo:
            b = yolo_boxes[yi]
            px1, py1, px2, py2 = b.pixel(W, H)
            cv2.rectangle(out, (px1, py1), (px2, py2), (60, 180, 60), 1)

        # ── Flash-анимация новых аномалий ─────────────────────────────
        now = time.time()
        self._flashes = [(fb, exp) for fb, exp in self._flashes
                         if exp > now]
        if self._flashes:
            alpha = 0.45 + 0.45 * abs(math.sin(now * 9))
            for fb, _ in self._flashes:
                px1, py1, px2, py2 = fb.pixel(W, H)
                ov = out.copy()
                cv2.rectangle(ov, (px1, py1), (px2, py2), C_RED, -1)
                cv2.addWeighted(ov, alpha * 0.28, out, 1 - alpha * 0.28,
                                0, out)

        # ── Баннер с баллами ──────────────────────────────────────────
        banner(out,
               f"🔴  YOUKAI АНОМАЛИИ  |  {fairy_event}  |  "
               f"Найдено: {self.caught_total}  |  "
               f"🔮 Youkai★: {self.youkai_score}  vs  ✨ Fairy★: {self.fairy_score}",
               bg=(70, 0, 0), fg=(255, 80, 80))

        return out, caught

    # ──────────────────────────────────────────────────────────────────
    def _log(self, msg: str) -> None:
        self.event_log.append(msg)
        if len(self.event_log) > 300:
            self.event_log.pop(0)

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _draw_waiting(frame: np.ndarray, prec: float) -> None:
        """Оверлей пока Youkai ещё не обучен достаточно."""
        H, W = frame.shape[:2]
        ov = np.zeros_like(frame)
        ov[:] = (8, 0, 16)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

        lines = [
            (f"Youkai обучается...", 0.90, (175, 70, 255)),
            (f"Precision: {prec:.1%} / порог {Y_ACTIVATE_PREC:.0%}",
             0.72, (200, 195, 100)),
            ("Panel 4 включится автоматически", 0.58, (120, 120, 190)),
        ]
        for i, (txt, sc, col) in enumerate(lines):
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
            cx = (W - tw) // 2
            cv2.putText(frame, txt, (cx, H//2 - 30 + i*48),
                        cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2)

        # Прогресс-бар
        bw = int(W * 0.68)
        bx = (W - bw) // 2
        by = H//2 + 90
        cv2.rectangle(frame, (bx, by), (bx+bw, by+16), (35, 0, 55), -1)
        fill = int(bw * min(1.0, prec / max(Y_ACTIVATE_PREC, 0.01)))
        if fill > 0:
            cv2.rectangle(frame, (bx, by), (bx+fill, by+16),
                          (190, 45, 255), -1)
        cv2.rectangle(frame, (bx, by), (bx+bw, by+16), (110, 70, 170), 1)


# ════════════════════════════════════════════════════════════════════════
#  РАЗДЕЛ 8 — ГЛАВНОЕ ПРИЛОЖЕНИЕ
# ════════════════════════════════════════════════════════════════════════

class App:
    """
    Главное окно приложения с 5 видео-панелями.

    Макет (1800×1020):
    ┌──────────────────────────────────────────────────────────┐
    │  Header (заголовок)                                      │
    ├──────────────────────────────────────────────────────────┤
    │  Controls (кнопки)                                       │
    ├────────────┬────────────┬────────────┬───────────────────┤
    │  Panel 0   │  Panel 1   │  Panel 2   │  Графики          │
    │  Исходник  │  YOLO 🟢   │  Fairy ⚡  │  (обе строки)     │
    ├────────────┼────────────┼────────────┤                   │
    │  Panel 3   │  Panel 4   │  Log 📋    │                   │
    │  Youkai 🔮 │  Аномалии🔴│            │                   │
    └────────────┴────────────┴────────────┴───────────────────┘
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(
            "🎯 FAIRY (YOLOv8) + YOUKAI  —  5 панелей детекции и аномалий"
        )
        self.root.geometry(f"{UI_W}x{UI_H}")
        self.root.configure(bg='#0b0b18')
        self.root.resizable(True, True)

        # ── Инициализация компонентов ─────────────────────────────────
        print("\n📹 Подготовка видео...")
        self.vid_path = get_video()

        print("\n🤖 YOLO (Fairy) — загрузка...")
        self.fairy      = FairyYOLO(YOLO_MODEL)

        print("\n🎭 Fairy Distorter — инициализация...")
        self.distorter  = FairyDistorter()

        print("\n🔮 Youkai — инициализация...")
        self.trainer    = YoukaiTrainer()
        self.trainer.load()   # Загружаем если есть чекпоинт

        print("\n🕵️ Anomaly Engine — инициализация...")
        self.anomaly    = AnomalyEngine()
        self.anomaly._log("⏳ Система запущена, ожидаю обучения Youkai...")

        self.state   = AppState()
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)

        self._setup_ui()

        # ── Потоки ───────────────────────────────────────────────────
        threading.Thread(target=self._train_loop, daemon=True,
                         name='Train').start()
        self._start_video()

        print(f"\n✅ Готово! Видео: {self.vid_path}")
        print(f"   Panel 4 активируется при precision ≥ {Y_ACTIVATE_PREC:.0%}\n")

    # ──────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        # ── Заголовок ─────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg='#0b0b18', height=70)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(
            hdr,
            text="🎯  FAIRY = YOLOv8  +  YOUKAI  —  5 панелей детекции и аномалий",
            font=('Arial', 17, 'bold'), bg='#0b0b18', fg='#00e5ff'
        ).pack(pady=3)
        tk.Label(
            hdr,
            text="P0 Исходник │ P1 YOLO→ЗЕЛЁНЫЙ │ P2 Fairy СКРЫВАЕТ/ДУБЛИРУЕТ │"
                 " P3 Youkai→ФИОЛЕТОВЫЙ │ P4 Аномалии→КРАСНЫЙ",
            font=('Arial', 8), bg='#0b0b18', fg='#546e7a'
        ).pack()

        # ── Панель управления ─────────────────────────────────────────
        ctrl = tk.Frame(self.root, bg='#10101e', height=50)
        ctrl.pack(fill=tk.X, padx=6, pady=2)
        ctrl.pack_propagate(False)

        kw  = dict(bg='#00e5ff', fg='#0b0b18', font=('Arial', 9, 'bold'),
                   relief='raised', bd=2, padx=9, pady=3,
                   activebackground='#00b0cc', cursor='hand2')
        kw2 = {**kw, 'bg': '#ff4444', 'fg': 'white',
                   'activebackground': '#cc1111'}

        tk.Button(ctrl, text="▶ СТАРТ",       command=self._start_video, **kw ).pack(side=tk.LEFT, padx=3, pady=7)
        tk.Button(ctrl, text="⏹ СТОП",        command=self._stop_video,  **kw ).pack(side=tk.LEFT, padx=3, pady=7)
        tk.Button(ctrl, text="💾 СОХРАНИТЬ",   command=self._save,        **kw ).pack(side=tk.LEFT, padx=3, pady=7)
        tk.Button(ctrl, text="📂 ЗАГРУЗИТЬ",   command=self._load,        **kw ).pack(side=tk.LEFT, padx=3, pady=7)
        tk.Button(ctrl, text="🔄 СБРОС",       command=self._reset,       **kw2).pack(side=tk.LEFT, padx=3, pady=7)

        sep = tk.Label(ctrl, text=" │ ", bg='#10101e', fg='#263238',
                       font=('Arial', 14))
        sep.pack(side=tk.LEFT, padx=3)

        self._v_event  = tk.StringVar(value="Fairy: BLACKOUT")
        self._v_status = tk.StringVar(value="🔄 Запуск...")
        self._v_caught = tk.StringVar(value="Youkai★:0 vs Fairy★:0")

        tk.Label(ctrl, textvariable=self._v_event,
                 bg='#10101e', fg='#ffb74d',
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=4)
        tk.Label(ctrl, textvariable=self._v_status,
                 bg='#10101e', fg='#69f0ae',
                 font=('Arial', 9,  'bold')).pack(side=tk.LEFT, padx=4)
        tk.Label(ctrl, textvariable=self._v_caught,
                 bg='#10101e', fg='#ff5252',
                 font=('Arial', 10, 'bold')).pack(side=tk.RIGHT, padx=12)

        # ── Основная область ──────────────────────────────────────────
        main = tk.Frame(self.root, bg='#0b0b18')
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        row1 = tk.Frame(main, bg='#0b0b18')
        row1.pack(fill=tk.BOTH, expand=True)
        row2 = tk.Frame(main, bg='#0b0b18')
        row2.pack(fill=tk.BOTH, expand=True, pady=3)

        def vpanel(parent, title, col):
            fr = tk.LabelFrame(parent, text=title, bg='#121224',
                               fg=col, font=('Arial', 9, 'bold'),
                               relief='ridge', bd=2)
            fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            fr.config(width=300, height=300)
            fr.pack_propagate(False)
            lb = tk.Label(fr, bg='black')
            lb.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            return lb

        # Строка 1: P0, P1, P2
        self.lbl = {}
        self.lbl[0] = vpanel(row1, "📹  P0 — ИСХОДНИК",               'white')
        self.lbl[1] = vpanel(row1, "🟢  P1 — FAIRY = YOLOv8 ЗЕЛЁНЫЙ", '#69f0ae')
        self.lbl[2] = vpanel(row1, "⚡  P2 — FAIRY ИСКАЖЕНИЯ",         '#ffb74d')

        # Графики — правый столбец строки 1 (занимает обе строки)
        gfr = tk.LabelFrame(row1, text="📊  ОБУЧЕНИЕ YOUKAI",
                             bg='#121224', fg='white',
                             font=('Arial', 9, 'bold'),
                             relief='ridge', bd=2)
        gfr.pack(side=tk.LEFT, fill=tk.Y, expand=False,
                 padx=2)
        gfr.config(width=340)
        gfr.pack_propagate(False)
        self._build_graphs(gfr)

        # Строка 2: P3, P4, Log
        self.lbl[3] = vpanel(row2, "🔮  P3 — YOUKAI ФИОЛЕТОВЫЙ",     '#ce93d8')
        self.lbl[4] = vpanel(row2, "🔴  P4 — ОБНАРУЖЕНИЕ АНОМАЛИЙ",  '#ef9a9a')

        log_fr = tk.LabelFrame(row2, text="📋  ЛОГ АНОМАЛИЙ",
                               bg='#121224', fg='#ff8a65',
                               font=('Arial', 9, 'bold'),
                               relief='ridge', bd=2)
        log_fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        self._log_txt = tk.Text(log_fr, bg='#08080f', fg='#ff5252',
                                font=('Consolas', 8), relief='flat',
                                state=tk.DISABLED, wrap=tk.WORD)
        sb = tk.Scrollbar(log_fr, command=self._log_txt.yview)
        self._log_txt.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_txt.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Placeholder
        ph = ImageTk.PhotoImage(Image.new('RGB', (PANEL_W, PANEL_H), (15, 15, 25)))
        for lb in self.lbl.values():
            lb.config(image=ph); lb.image = ph

    # ──────────────────────────────────────────────────────────────────
    def _build_graphs(self, parent: tk.Frame):
        """Создаёт 4 графика статистики."""
        fig, axs = plt.subplots(4, 1, figsize=(3.3, 7.8))
        fig.patch.set_facecolor('#0b0b18')
        fig.tight_layout(pad=1.8, h_pad=1.4)
        self._fig = fig

        configs = [
            ('Precision Youkai', '#69f0ae'),
            ('Loss Youkai',      '#ff6b6b'),
            ('YOLO чел./кадр',   '#ffd740'),
            ('Итого аномалий',   '#ff5252'),
        ]
        self._glines: Dict[str, object] = {}
        self._gaxes:  Dict[str, object] = {}

        for ax, (title, col) in zip(axs, configs):
            ax.set_title(title, color='white', fontsize=8, pad=2)
            ax.set_facecolor('#0e0e20')
            ax.tick_params(colors='#546e7a', labelsize=6)
            ax.grid(True, alpha=0.18, color='#263238')
            for sp in ax.spines.values():
                sp.set_color('#1c2a33')
            line, = ax.plot([], [], col, lw=1.2)
            self._glines[title] = line
            self._gaxes[title]  = ax

        # Порог Panel 4
        self._gaxes['Precision Youkai'].set_ylim(0, 1)
        self._gaxes['Precision Youkai'].axhline(
            y=Y_ACTIVATE_PREC, color='#ffff00', lw=0.8, ls='--', alpha=0.7
        )

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._gcanvas = canvas

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _to_photo(arr: np.ndarray, size: Tuple[int,int]) -> ImageTk.PhotoImage:
        img = Image.fromarray(arr)
        img.thumbnail(size, Image.LANCZOS)
        canvas = Image.new('RGB', size, (15, 15, 25))
        canvas.paste(img, ((size[0]-img.size[0])//2,
                           (size[1]-img.size[1])//2))
        return ImageTk.PhotoImage(canvas)

    def _put(self, panel_id: int, arr: np.ndarray) -> None:
        """Обновляет одну панель в главном потоке Tkinter."""
        lb = self.lbl[panel_id]
        w  = max(PANEL_W, lb.winfo_width())
        h  = max(PANEL_H, lb.winfo_height())
        ph = self._to_photo(arr, (w, h))
        lb.config(image=ph); lb.image = ph

    # ──────────────────────────────────────────────────────────────────
    def _start_video(self):
        self._stop_video()
        self.running = True
        threading.Thread(target=self._video_loop, daemon=True,
                         name='Video').start()
        self._v_status.set("🎬 Запущено | Youkai обучается...")

    def _stop_video(self):
        self.running = False
        time.sleep(0.12)
        self._v_status.set("⏸ Остановлено")

    def _save(self):
        p = filedialog.asksaveasfilename(
            defaultextension='.pth',
            filetypes=[('PyTorch', '*.pth'), ('Все', '*.*')],
            initialfile='youkai_v7.pth', title='Сохранить Youkai'
        )
        if p and self.trainer.save(p):
            self._v_status.set(f"💾 {os.path.basename(p)}")
            messagebox.showinfo("Сохранение",
                                f"Youkai сохранён!\n"
                                f"Шагов: {self.trainer.step}\n"
                                f"Best precision: {self.trainer.best_prec:.1%}")

    def _load(self):
        p = filedialog.askopenfilename(
            filetypes=[('PyTorch', '*.pth'), ('Все', '*.*')]
        )
        if p and self.trainer.load(p):
            self._v_status.set(f"📂 {os.path.basename(p)}")
            messagebox.showinfo("Загрузка",
                                f"Youkai загружен!\n"
                                f"Шагов: {self.trainer.step}\n"
                                f"Best precision: {self.trainer.best_prec:.1%}")

    def _reset(self):
        if messagebox.askyesno("Сброс",
                               "Полностью сбросить обучение Youkai?"):
            self.trainer = YoukaiTrainer()
            self.anomaly = AnomalyEngine()
            self._v_status.set("🔄 Сброс выполнен!")
            self._v_caught.set("Youkai★:0 vs Fairy★:0")

    # ──────────────────────────────────────────────────────────────────
    # ПОТОК ОТОБРАЖЕНИЯ
    # ──────────────────────────────────────────────────────────────────
    def _video_loop(self):
        """
        Читает кадры из видео и передаёт их в очередь для _train_loop.
        Также обновляет все 5 панелей на экране.
        """
        cap = cv2.VideoCapture(self.vid_path)
        if not cap.isOpened():
            self.vid_path = _make_synthetic_video()
            cap = cv2.VideoCapture(self.vid_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        fn = 0

        while self.running:
            try:
                ret, bgr = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.02)
                    continue

                # Передаём кадр в очередь для _train_loop
                try:
                    self.frame_queue.put_nowait(bgr)
                except:
                    pass  # очередь переполнена, пропускаем

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # Получаем готовые кадры из состояния
                with self.state.lock:
                    panels = {i: self.state.panel[i] for i in range(5)}
                    yb     = list(self.state.yolo_boxes)
                    ev     = self.state.fairy_event
                    prec   = self.state.youkai_prec
                    step   = self.state.youkai_step
                    caught = self.anomaly.caught_total
                    youkai_score = self.anomaly.youkai_score
                    fairy_score  = self.anomaly.fairy_score

                # Panel 0 — всегда актуальный исходник
                self._put(0, rgb)

                # Panels 1-4 — из состояния
                for i in range(1, 5):
                    if panels[i] is not None:
                        self._put(i, panels[i])

                # Строка статуса
                self._v_event.set(
                    f"Fairy: {'⬛ BLACKOUT' if ev == 'BLACKOUT' else '👥 DUPLICATE'}"
                )
                self._v_caught.set(f"Youkai★:{youkai_score} vs Fairy★:{fairy_score}")
                p4_ok = prec >= Y_ACTIVATE_PREC
                self._v_status.set(
                    f"🎯 Шаг:{step}  Prec:{prec:.1%}  "
                    f"YOLO:{len(yb)}чел  "
                    f"{'🔴P4 ON' if p4_ok else '⏳P4 OFF'}"
                )

                # Графики — каждые 20 кадров
                if fn % 20 == 0:
                    self._update_graphs()

                # Лог — каждые 12 кадров
                if fn % 12 == 0:
                    self._update_log()

                fn += 1
                time.sleep(0.032)

            except Exception as e:
                print(f"[Video] {e}")
                traceback.print_exc()
                time.sleep(0.1)

        cap.release()

    # ──────────────────────────────────────────────────────────────────
    # ОБУЧАЮЩИЙ ПОТОК
    # ──────────────────────────────────────────────────────────────────
    def _train_loop(self):
        """
        Вычислительный поток — получает кадры из очереди (от _video_loop)
        и обучает модель.
        """
        while True:
            try:
                # Пытаемся получить кадр из очереди с таймаутом
                try:
                    bgr = self.frame_queue.get(timeout=0.1)
                except:
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # ── 1. YOLO: детектируем людей ─────────────────────────
                yolo_boxes = self.fairy.detect(bgr)

                # ── 2. Panel 1: зелёная заливка людей ─────────────────
                p1 = FairyYOLO.render_panel1(rgb, yolo_boxes)

                # ── 3. Panel 2: Fairy искажает кадр ────────────────────
                distorted, fairy_anoms = self.distorter.apply(rgb, yolo_boxes)

                # ── 4. Обучаем Youkai на GT от YOLO ───────────────────
                frame_t = to_tensor(rgb)
                loss, prec, rec = self.trainer.train_step(frame_t, yolo_boxes)

                # ── 5. Panel 3: Youkai рисует фиолетовых людей ─────────
                if self.trainer.step % 4 == 0:
                    yk_boxes = self.trainer.detect(frame_t)
                else:
                    with self.state.lock:
                        yk_boxes = list(self.state.youkai_boxes)

                p3 = YoukaiTrainer.render_panel3(rgb, yk_boxes,
                                                  self.trainer.step, prec)

                # ── 6. Panel 4: Anomaly Engine ─────────────────────────
                p4, caught = self.anomaly.analyze(
                    distorted,
                    self.trainer,
                    yolo_boxes,
                    fairy_anoms,
                    self.distorter.event,
                    prec
                )

                # ── 7. Запись в AppState ───────────────────────────────
                with self.state.lock:
                    self.state.panel[1]    = p1
                    self.state.panel[2]    = distorted
                    self.state.panel[3]    = p3
                    self.state.panel[4]    = p4
                    self.state.yolo_boxes  = yolo_boxes
                    self.state.youkai_boxes= yk_boxes
                    self.state.fairy_event = self.distorter.event
                    self.state.youkai_prec = prec
                    self.state.youkai_rec  = rec
                    self.state.youkai_step = self.trainer.step
                    self.state.caught_total= self.anomaly.caught_total
                    self.state.youkai_score= self.anomaly.youkai_score
                    self.state.fairy_score = self.anomaly.fairy_score
                    self.state.h_prec.append(prec)
                    self.state.h_loss.append(loss)
                    self.state.h_yolo.append(len(yolo_boxes))

                # ── Консольный лог ─────────────────────────────────────
                if self.trainer.step % LOG_STEPS == 0:
                    ap = (float(np.mean(list(self.trainer.h_prec)[-50:]))
                          if self.trainer.h_prec else 0.0)
                    print(
                        f"  [#{self.trainer.step:5d}] "
                        f"loss={loss:.4f} "
                        f"prec={prec:.1%} "
                        f"avg50={ap:.1%} "
                        f"yolo={len(yolo_boxes)} "
                        f"event={self.distorter.event} "
                        f"caught={self.anomaly.caught_total}"
                    )

                time.sleep(0.001)

            except Exception as e:
                print(f"[Train] {e}")
                traceback.print_exc()
                time.sleep(0.2)

    # ──────────────────────────────────────────────────────────────────
    # ОБНОВЛЕНИЕ ГРАФИКОВ И ЛОГА
    # ──────────────────────────────────────────────────────────────────
    def _update_graphs(self):
        try:
            with self.state.lock:
                ph = list(self.state.h_prec)
                lo = list(self.state.h_loss)
                yh = list(self.state.h_yolo)

            if len(ph) < 2:
                return

            xs = list(range(len(ph)))

            def _upd(name, x, y, auto=True):
                line = self._glines[name]
                ax   = self._gaxes[name]
                line.set_data(x, y)
                ax.set_xlim(0, max(60, len(x)))
                if auto:
                    ax.relim(); ax.autoscale_view(scalex=False)

            _upd('Precision Youkai', xs, ph, auto=False)
            _upd('Loss Youkai',      xs, lo)
            _upd('YOLO чел./кадр',   xs, yh[:len(xs)])

            # Сводная панель
            ax4  = self._gaxes['Итого аномалий']
            ax4.clear(); ax4.set_facecolor('#0e0e20'); ax4.axis('off')
            ax4.set_title('Итого аномалий', color='white', fontsize=8, pad=2)

            step = self.trainer.step
            ap   = float(np.mean(ph[-50:])) if ph else 0.0
            p4   = ap >= Y_ACTIVATE_PREC

            ax4.text(0.5, 0.68, str(self.anomaly.caught_total),
                     ha='center', transform=ax4.transAxes,
                     fontsize=28, color='#ff5252', fontweight='bold')
            ax4.text(0.5, 0.40,
                     f"Шаг: {step}   Prec: {ap:.1%}",
                     ha='center', transform=ax4.transAxes,
                     fontsize=8, color='#b0bec5')
            p4txt = "Panel 4: ВКЛ ✓" if p4 else "Panel 4: ждём..."
            p4col = '#69f0ae' if p4 else '#ffb74d'
            ax4.text(0.5, 0.16, p4txt,
                     ha='center', transform=ax4.transAxes,
                     fontsize=8, color=p4col)

            self._gcanvas.draw_idle()

        except Exception:
            pass

    def _update_log(self):
        try:
            lines = self.anomaly.event_log[-80:]
            self._log_txt.config(state=tk.NORMAL)
            self._log_txt.delete('1.0', tk.END)
            for ln in reversed(lines):
                self._log_txt.insert(tk.END, ln + '\n')
            self._log_txt.config(state=tk.DISABLED)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.running = False
        if messagebox.askyesno(
            "Выход",
            f"Сохранить Youkai перед выходом?\n\n"
            f"Шагов: {self.trainer.step}\n"
            f"Best precision: {self.trainer.best_prec:.1%}"
        ):
            p = filedialog.asksaveasfilename(
                defaultextension='.pth',
                filetypes=[('PyTorch', '*.pth'), ('Все', '*.*')],
                initialfile='youkai_v7.pth'
            )
            if p:
                self.trainer.save(p)
        self.root.destroy()


# ════════════════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if not YOLO_AVAILABLE:
        print("\n" + "=" * 55)
        print("❌  ultralytics не установлен!")
        print("    Запустите: pip install ultralytics")
        print("    Затем снова: python FairyYoukai_7v.py")
        print("=" * 55)
        raise SystemExit(1)

    print("\n" + "=" * 64)
    print("  Panel 0  → Исходное видео")
    print("  Panel 1  → Fairy=YOLO: закрашивает людей ЗЕЛЁНЫМ")
    print("  Panel 2  → Fairy Distort: СКРЫВАЕТ или ДУБЛИРУЕТ людей")
    print(f"             (переключается каждые {FAIRY_EVENT_SEC:.1f}с)")
    print("  Panel 3  → Youkai учится → закрашивает людей ФИОЛЕТОВЫМ")
    print("  Panel 4  → Youkai ловит аномалии → КРАСНЫЕ рамки")
    print(f"             (активируется при precision ≥ {Y_ACTIVATE_PREC:.0%})")
    print()
    print(f"  YOLO    : {YOLO_MODEL}")
    print(f"  Youkai  : {Y_ANCHORS} якорей, вход {Y_IN_W}×{Y_IN_H}")
    print(f"  Устройство: {DEVICE}")
    print("=" * 64 + "\n")

    App().run()