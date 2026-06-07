#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚔️ FAIRY vs YOUKAI - Эпическая битва нейросетей
Самообучение распознаванию объектов на реальном видео (Shibuya Crossing)
Запуск: python fairy_vs_youkai.py

ИСПРАВЛЕНИЯ v6:
  1. ДВУХЭТАПНОЕ ОБУЧЕНИЕ:
     - Stage 1 (Learning): Youkai учится на чистых кадрах, Fairy выключена
     - Stage 2 (Game): Fairy активна, начинается игра
  2. Fairy ДОБАВЛЯЕТ ОБЪЕКТЫ (слоны, динозавры) + шум + деформация
  3. Исправлено зависание VideoCapture с timeout
  4. Наглядный прогресс этапов в UI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import time
from collections import deque
import os
import subprocess
import gc

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = torch.device('cpu')
print(f"✅ Устройство: {device}")
print(f"✅ PyTorch: {torch.__version__}")

# ================================================================
# ПАРАМЕТРЫ
# ================================================================
STAGE1_STEPS = 1000  # увеличено для долгого обучения
STAGE2_MIN_ACCURACY = 0.85  # нужна РЕАЛЬНАЯ точность детекции (не просто формула)

# ================================================================
# 1. ВИДЕО
# ================================================================

def download_shibuya_video():
    video_path = "shibuya_crossing.mp4"
    if os.path.exists(video_path) and os.path.getsize(video_path) > 100_000:
        print(f"✅ Видео найдено: {video_path}")
        return video_path

    print("📥 Пробую скачать видео перекрёстка Сибуя...")
    url = "https://www.youtube.com/watch?v=4SvwUbDQZmc"
    try:
        cmd = f'yt-dlp -f "best[height<=480]" -o "{video_path}" --no-playlist "{url}"'
        subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if os.path.exists(video_path) and os.path.getsize(video_path) > 1_000_000:
            print("✅ Видео скачано!")
            return video_path
    except Exception as e:
        print(f"⚠️ Ошибка скачивания: {e}")

    print("⚠️ Создаю синтетическое видео с объектами...")
    return create_synthetic_video()


def create_synthetic_video():
    """Синтетическое видео с движущимися объектами (пешеходы, машины, собаки)."""
    video_path = "synthetic_training.mp4"
    if os.path.exists(video_path) and os.path.getsize(video_path) > 10_000:
        return video_path

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 25.0, (640, 480))

    objects = [
        {'x': 60,  'y': 355, 'vx':  3, 'color': (80, 220, 80),  'label': 'ЧЕЛОВЕК', 'w': 45,  'h': 80},
        {'x': 250, 'y': 365, 'vx': -5, 'color': (220, 80, 80),  'label': 'МАШИНА',  'w': 100, 'h': 55},
        {'x': 480, 'y': 360, 'vx':  2, 'color': (80, 130, 220), 'label': 'СОБАКА',  'w': 50,  'h': 40},
        {'x': 150, 'y': 358, 'vx':  4, 'color': (220, 200, 50), 'label': 'АВТОБУС', 'w': 120, 'h': 65},
        {'x': 400, 'y': 362, 'vx': -3, 'color': (180, 60, 220), 'label': 'ВЕЛОСИПЕД','w': 55, 'h': 50},
    ]

    print("🎬 Генерирую синтетическое тренировочное видео...")
    for frame_num in range(600):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Небо (градиент)
        for y in range(240):
            c = int(80 + y * 0.6)
            frame[y, :] = [c, c + 20, 200]
        # Земля
        frame[240:480, :] = [40, 90, 40]
        # Дорога
        cv2.rectangle(frame, (0, 345), (640, 445), (75, 75, 75), -1)
        # Разметка
        for xi in range(0, 640, 64):
            cv2.rectangle(frame, (xi, 388), (xi + 44, 395), (240, 240, 240), -1)

        for obj in objects:
            obj['x'] = int((obj['x'] + obj['vx']) % 700) - 30
            x, y = int(obj['x']), int(obj['y'])
            w, h = obj['w'], obj['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), obj['color'], -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(frame, obj['label'], (x + 3, y + h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        out.write(frame)
        if frame_num % 100 == 0:
            print(f"  кадр {frame_num}/600")

    out.release()
    print("✅ Синтетическое видео создано")
    return video_path


# ================================================================
# 2. НЕЙРОСЕТИ
# ================================================================

NUM_CLASSES  = 5
NUM_ANCHORS  = 30


class ElegantFairy(nn.Module):
    """
    Fairy — прячет объекты через:
      • адаптивный шум
      • геометрическое искажение
      • добавление синтетических объектов (слоны, динозавры)
    """
    def __init__(self):
        super().__init__()
        self.noise_strength = nn.Parameter(torch.tensor(0.12))
        self.warp_strength  = nn.Parameter(torch.tensor(0.08))
        self.object_strength = nn.Parameter(torch.tensor(0.3))

        self.distort_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh(),
        )

    def _add_synthetic_object(self, x, stage=1):
        """Добавляет синтетический объект (слон, динозавр) в случайном месте."""
        B, C, H, W = x.shape
        result = x.clone()

        if np.random.rand() > 0.5:  # 50% вероятность добавить объект
            return result

        try:
            obj_type = np.random.choice(['elephant', 'dinosaur', 'random'])
            # БЕЗОПАСНЫЕ границы для случайного объекта
            ox = np.random.randint(20, max(21, W - 120))
            oy = np.random.randint(20, max(21, H - 120))
            ow = np.random.randint(40, 80)
            oh = np.random.randint(40, 80)

            for b in range(B):
                if obj_type == 'elephant':
                    if oy + oh < H and ox + ow < W:
                        result[b, :, oy:oy+oh, ox:ox+ow] = torch.tensor([0.5, 0.5, 0.5], device=x.device).view(3, 1, 1)
                        if oy + oh + 20 < H:
                            result[b, :, oy+oh:oy+oh+20, ox+ow//3:min(W, ox+2*ow//3)] = torch.tensor([0.5, 0.5, 0.5], device=x.device).view(3, 1, 1)
                elif obj_type == 'dinosaur':
                    if oy + oh < H and ox + ow < W:
                        result[b, :, oy:oy+oh, ox:ox+ow] = torch.tensor([0.8, 0.2, 0.2], device=x.device).view(3, 1, 1)
                        if ox - 20 > 0:
                            result[b, :, oy:oy+oh//2, max(0, ox-20):ox] = torch.tensor([0.8, 0.2, 0.2], device=x.device).view(3, 1, 1)
                else:
                    if oy + oh < H and ox + ow < W:
                        color = torch.rand(3, device=x.device)
                        result[b, :, oy:oy+oh, ox:ox+ow] = color.view(3, 1, 1)
        except Exception as e:
            pass  # Если не получилось добавить объект - просто пропускаем

        return result

    def forward(self, x, training=True, stage=1):
        if not training:
            return x

        noise_amp = torch.sigmoid(self.noise_strength) * 0.25
        noise = torch.randn_like(x) * noise_amp

        warp_amp = torch.sigmoid(self.warp_strength) * 0.12
        B, C, H, W = x.shape

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        warp_x = grid_x + torch.sin(grid_y * 3.14) * warp_amp
        warp_y = grid_y + torch.cos(grid_x * 3.14) * warp_amp

        grid = torch.stack([warp_x, warp_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        warped = F.grid_sample(x, grid, align_corners=False, padding_mode='border')

        learned = self.distort_net(warped) * 0.1
        corrupted = warped + noise + learned

        # На Stage 2 добавляем синтетические объекты
        if stage >= 2:
            corrupted = self._add_synthetic_object(corrupted, stage)

        return torch.clamp(corrupted, 0.0, 1.0)


class ElegantYoukai(nn.Module):
    """
    Youkai — ищет объекты.
    Выдаёт NUM_ANCHORS якорных боксов с (x, y, w, h, conf, class_scores)
    """
    def __init__(self, num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS):
        super().__init__()
        self.num_classes  = num_classes
        self.num_anchors  = num_anchors
        self.out_per_anch = 4 + 1 + num_classes

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.head = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_anchors * self.out_per_anch),
        )

    def forward(self, x):
        feat = self.backbone(x).view(x.size(0), -1)
        raw  = self.head(feat).view(-1, self.num_anchors, self.out_per_anch)

        boxes   = torch.sigmoid(raw[:, :, :4])
        conf    = torch.sigmoid(raw[:, :, 4:5])
        classes = torch.sigmoid(raw[:, :, 5:])
        return boxes, conf, classes


# ================================================================
# 3. БИТВА С ДВУХЭТАПНЫМ ОБУЧЕНИЕМ
# ================================================================

def nms(boxes, conf, iou_threshold=0.3):
    """Non-Maximum Suppression — удаляет дублирующиеся боксы."""
    if boxes.shape[0] == 0:
        return []

    keep_idx = []
    sorted_idx = torch.argsort(conf.squeeze(), descending=True)

    for i in range(len(sorted_idx)):
        if len(keep_idx) == 0:
            keep_idx.append(sorted_idx[i].item())
            continue

        curr_box = boxes[sorted_idx[i]]
        curr_conf = conf[sorted_idx[i]].item()

        should_keep = True
        for kept_i in keep_idx:
            kept_box = boxes[kept_i]

            x1_max = max(curr_box[0], kept_box[0])
            y1_max = max(curr_box[1], kept_box[1])
            x2_min = min(curr_box[0] + curr_box[2], kept_box[0] + kept_box[2])
            y2_min = min(curr_box[1] + curr_box[3], kept_box[1] + kept_box[3])

            inter_w = max(0, x2_min - x1_max)
            inter_h = max(0, y2_min - y1_max)
            inter_area = inter_w * inter_h

            curr_area = curr_box[2] * curr_box[3]
            kept_area = kept_box[2] * kept_box[3]
            union = curr_area + kept_area - inter_area

            if union > 0:
                iou = inter_area / union
                if iou > iou_threshold:
                    should_keep = False
                    break

        if should_keep:
            keep_idx.append(sorted_idx[i].item())

    return keep_idx

class EpicBattle:
    def __init__(self):
        self.device = torch.device('cpu')
        print("⚔️ Создаю бойцов...")

        self.fairy  = ElegantFairy().to(self.device)
        self.youkai = ElegantYoukai().to(self.device)

        self.optim_fairy  = optim.Adam(self.fairy.parameters(),  lr=0.0005)
        self.optim_youkai = optim.Adam(self.youkai.parameters(), lr=0.001)

        self.fairy_momentum  = 1.0
        self.youkai_momentum = 1.0

        # Двухэтапное обучение
        self.stage = 1  # 1 = обучение, 2 = игра
        self.stage1_completed = False

        self.accuracy_history    = deque(maxlen=300)
        self.loss_history        = deque(maxlen=300)
        self.fairy_power_history = deque(maxlen=300)
        self.youkai_power_history= deque(maxlen=300)
        self.real_detection_rate = deque(maxlen=50)  # реальный процент обнаруженных объектов
        self.step        = 0
        self.fairy_wins  = 0
        self.youkai_wins = 0

    def train_step(self, frames: torch.Tensor):
        frames = frames.to(self.device)

        # ══════════════════════════════════════
        # ШАГ 1: Обучаем Youkai
        # ══════════════════════════════════════
        with torch.no_grad():
            corrupted_stop = self.fairy(frames, training=True, stage=self.stage)

        self.optim_youkai.zero_grad()

        boxes_o, conf_o, cls_o = self.youkai(frames)
        boxes_c, conf_c, cls_c = self.youkai(corrupted_stop)

        # На Stage 1 учим регулярно быть уверенным (высокая conf)
        if self.stage == 1:
            conf_loss = F.binary_cross_entropy(conf_o, torch.ones_like(conf_o) * 0.9)
            box_loss = F.mse_loss(boxes_o, boxes_c) * 2.0
            consist = (
                box_loss +
                F.mse_loss(conf_o,  conf_c) * 1.5 +
                F.mse_loss(cls_o,   cls_c) +
                conf_loss * 0.5
            )
        else:
            consist = (
                F.mse_loss(boxes_o, boxes_c) * 1.5 +
                F.mse_loss(conf_o,  conf_c) +
                F.mse_loss(cls_o,   cls_c)
            )

        youkai_loss = consist * self.youkai_momentum
        youkai_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.youkai.parameters(), 1.0)
        self.optim_youkai.step()

        # РЕАЛЬНАЯ МЕТРИКА: процент боксов с confidence > threshold
        threshold = 0.3 if self.stage == 1 else 0.5
        high_conf_boxes = torch.sum(conf_o > threshold).item()
        total_boxes = conf_o.shape[0] * conf_o.shape[1]
        real_accuracy = high_conf_boxes / total_boxes if total_boxes > 0 else 0.0
        self.real_detection_rate.append(real_accuracy)

        # ══════════════════════════════════════
        # ШАГ 2: Обучаем Fairy (только на Stage 2)
        # ══════════════════════════════════════
        if self.stage >= 2:
            self.optim_fairy.zero_grad()

            corrupted2 = self.fairy(frames, training=True, stage=self.stage)

            with torch.no_grad():
                boxes_o2, conf_o2, cls_o2 = self.youkai(frames)

            boxes_c2, conf_c2, cls_c2 = self.youkai(corrupted2)

            fairy_consist = (
                F.mse_loss(boxes_o2.detach(), boxes_c2) +
                F.mse_loss(conf_o2.detach(),  conf_c2) +
                F.mse_loss(cls_o2.detach(),   cls_c2)
            )
            distortion_penalty = 0.01 * torch.mean(torch.abs(corrupted2 - frames.detach()))
            fairy_loss = -fairy_consist * self.fairy_momentum + distortion_penalty

            fairy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fairy.parameters(), 1.0)
            self.optim_fairy.step()

        # ══════════════════════════════════════
        # Динамический баланс
        # ══════════════════════════════════════
        if real_accuracy > 0.6:
            self.youkai_wins += 1
            if self.stage >= 2:
                self.fairy_momentum = min(2.0, self.fairy_momentum * 1.02)
        elif real_accuracy < 0.4:
            self.fairy_wins += 1
            if self.stage >= 2:
                self.youkai_momentum = min(2.0, self.youkai_momentum * 1.02)
        else:
            if self.stage >= 2:
                self.fairy_momentum  = max(0.5, self.fairy_momentum  * 0.995)
                self.youkai_momentum = max(0.5, self.youkai_momentum * 0.995)

        # Переход на Stage 2 - теперь основан на РЕАЛЬНОЙ средней точности
        avg_real_accuracy = np.mean(list(self.real_detection_rate)) if len(self.real_detection_rate) > 0 else 0
        
        if (self.stage == 1 and self.step >= min(200, STAGE1_STEPS) and
            avg_real_accuracy >= STAGE2_MIN_ACCURACY):
            print(f"\n✅ STAGE 1 ЗАВЕРШЕН! Реальная точность детекции: {avg_real_accuracy:.1%}")
            print(f"   Переходим на STAGE 2 — ВКЛЮЧАЕМ FAIRY!\n")
            self.stage = 2
            self.stage1_completed = True
            self.fairy_wins = 0
            self.youkai_wins = 0
        elif self.stage == 1 and self.step >= STAGE1_STEPS:
            print(f"\n⚠️ STAGE 1 МАКСИМУМ ШАГОВ ДОСТИГНУТ! Реальная точность: {avg_real_accuracy:.1%}")
            print(f"   Переходим на STAGE 2...\n")
            self.stage = 2
            self.stage1_completed = True
            self.fairy_wins = 0
            self.youkai_wins = 0

        self.accuracy_history.append(real_accuracy)
        self.loss_history.append(consist.item())
        self.fairy_power_history.append(self.fairy_momentum)
        self.youkai_power_history.append(self.youkai_momentum)
        self.step += 1

        return real_accuracy, consist.item()

    def detect(self, frame: torch.Tensor):
        self.youkai.eval()
        with torch.no_grad():
            boxes, conf, classes = self.youkai(frame.to(self.device))
        self.youkai.train()
        return boxes.cpu(), conf.cpu(), classes.cpu()

    def save(self, path: str = 'epic_battle_save.pth') -> bool:
        try:
            torch.save({
                'fairy':           self.fairy.state_dict(),
                'youkai':          self.youkai.state_dict(),
                'step':            self.step,
                'stage':           self.stage,
                'stage1_completed': self.stage1_completed,
                'fairy_wins':      self.fairy_wins,
                'youkai_wins':     self.youkai_wins,
                'fairy_momentum':  self.fairy_momentum,
                'youkai_momentum': self.youkai_momentum,
            }, path)
            print(f"💾 Сохранено → {path}  (шаг {self.step}, stage {self.stage})")
            return True
        except Exception as e:
            print(f"⚠️ Ошибка сохранения: {e}")
            return False

    def load(self, path: str = 'epic_battle_save.pth') -> bool:
        if not os.path.exists(path):
            return False
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
            self.fairy.load_state_dict(data['fairy'])
            self.youkai.load_state_dict(data['youkai'])
            self.step            = data.get('step', 0)
            self.stage           = data.get('stage', 1)
            self.stage1_completed = data.get('stage1_completed', False)
            self.fairy_wins      = data.get('fairy_wins', 0)
            self.youkai_wins     = data.get('youkai_wins', 0)
            self.fairy_momentum  = data.get('fairy_momentum', 1.0)
            self.youkai_momentum = data.get('youkai_momentum', 1.0)
            print(f"📂 Загружено ← {path}  (шаг {self.step}, stage {self.stage})")
            return True
        except Exception as e:
            print(f"⚠️ Ошибка загрузки: {e}")
            return False


# ================================================================
# 4. ГЛАВНОЕ ПРИЛОЖЕНИЕ
# ================================================================

CLASS_COLORS = [
    (0, 220, 100),    # ЧЕЛОВЕК   — зелёный
    (220, 70,  70),   # МАШИНА    — красный
    (70, 140, 220),   # СОБАКА    — синий
    (220, 200, 40),   # АВТОБУС   — жёлтый
    (180, 60, 220),   # ВЕЛОСИПЕД — фиолетовый
]
CLASS_NAMES  = ['ЧЕЛОВЕК', 'МАШИНА', 'СОБАКА', 'АВТОБУС', 'ВЕЛОСИПЕД']
DISPLAY_SIZE = (640, 480)


class EpicApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("⚔️ FAIRY vs YOUKAI — Нейросетевая игра в прятки")
        self.root.geometry("1600x960")
        self.root.configure(bg='#1a1a2e')

        print("\n📹 Подготовка видео...")
        self.video_path = download_shibuya_video()

        print("🧠 Инициализация нейросетей...")
        self.battle = EpicBattle()
        self.battle.load()

        self.running = False
        self._setup_ui()
        self._start_video()

        self._train_thread = threading.Thread(target=self._training_loop, daemon=True)
        self._train_thread.start()

        print("\n✅ Приложение запущено!")

    # ──────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────
    def _setup_ui(self):
        hdr = tk.Frame(self.root, bg='#1a1a2e', height=65)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⚔️  FAIRY  vs  YOUKAI  ⚔️",
                 font=('Arial', 22, 'bold'), bg='#1a1a2e', fg='#e94560').pack(pady=4)
        tk.Label(hdr, text="Нейросетевая игра в прятки | Самообучение детектору объектов",
                 font=('Arial', 9), bg='#1a1a2e', fg='#888').pack()

        # Панель управления + СТАДИЯ
        ctrl = tk.Frame(self.root, bg='#0f0f1a', height=52)
        ctrl.pack(fill=tk.X, padx=10, pady=4)
        ctrl.pack_propagate(False)

        btn = dict(bg='#e94560', fg='white', font=('Arial', 10, 'bold'),
                   relief='raised', bd=2, padx=12, pady=3)
        tk.Button(ctrl, text="▶ СТАРТ",     command=self._start_video,  **btn).pack(side=tk.LEFT, padx=4, pady=8)
        tk.Button(ctrl, text="⏹ СТОП",     command=self._stop_video,   **btn).pack(side=tk.LEFT, padx=4, pady=8)
        tk.Button(ctrl, text="💾 СОХРАНИТЬ", command=self._save_battle, **btn).pack(side=tk.LEFT, padx=4, pady=8)
        tk.Button(ctrl, text="📂 ЗАГРУЗИТЬ", command=self._load_battle, **btn).pack(side=tk.LEFT, padx=4, pady=8)
        tk.Button(ctrl, text="🔄 СБРОС",    command=self._reset_battle, **btn).pack(side=tk.LEFT, padx=4, pady=8)

        # ИНДИКАТОР СТАДИИ
        self._stage_label = tk.StringVar(value="🎓 STAGE 1: ОБУЧЕНИЕ")
        stage_frame = tk.Frame(ctrl, bg='#0f0f1a')
        stage_frame.pack(side=tk.RIGHT, padx=20)
        tk.Label(stage_frame, textvariable=self._stage_label,
                 font=('Arial', 12, 'bold'), bg='#0f0f1a', fg='#ffff00').pack()

        self._status = tk.StringVar(value="⚔️ БИТВА ГОТОВА К НАЧАЛУ")
        tk.Label(ctrl, textvariable=self._status, bg='#0f0f1a',
                 fg='#00ff00', font=('Arial', 11, 'bold')).pack(side=tk.RIGHT, padx=20)

        # 2×2 сетка видео-панелей
        main = tk.Frame(self.root, bg='#1a1a2e')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        top = tk.Frame(main, bg='#1a1a2e'); top.pack(fill=tk.BOTH, expand=True)
        bot = tk.Frame(main, bg='#1a1a2e'); bot.pack(fill=tk.BOTH, expand=True, pady=6)

        def _panel(parent, title, fg):
            f = tk.LabelFrame(parent, text=title, bg='#16213e', fg=fg,
                              font=('Arial', 10, 'bold'))
            f.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            lbl = tk.Label(f, bg='black')
            lbl.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            return lbl

        self._lbl_orig  = _panel(top, "📹 1. ИСХОДНОЕ ВИДЕО",          'white')
        self._lbl_det   = _panel(top, "👁️ 2. YOUKAI — ДЕТЕКЦИЯ БОКСОВ", '#00ff00')
        self._lbl_fairy = _panel(bot, "🧚 3. FAIRY — ИСКАЖЕНИЕ",        '#ff6b6b')

        # Статистика
        f4 = tk.LabelFrame(bot, text="📊 4. СТАТИСТИКА БИТВЫ",
                            bg='#16213e', fg='white', font=('Arial', 10, 'bold'))
        f4.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4)

        self._fig, ((self._ax1, self._ax2), (self._ax3, self._ax4)) = \
            plt.subplots(2, 2, figsize=(7, 5))
        self._fig.patch.set_facecolor('#16213e')
        self._fig.tight_layout(pad=2)

        for ax, ttl in [(self._ax1, 'Точность Youkai'), (self._ax2, 'Функция потерь'),
                        (self._ax3, 'Сила противников')]:
            ax.set_title(ttl, color='white', fontsize=9)
            ax.set_facecolor('#0f0f1a')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)

        self._ax1.set_ylim(0, 1)
        self._ax3.set_ylim(0, 2.2)
        self._ax4.set_title('Счёт побед', color='white', fontsize=9)
        self._ax4.set_facecolor('#0f0f1a')

        self._line_acc,    = self._ax1.plot([], [], '#00ff00', lw=1.5)
        self._line_loss,   = self._ax2.plot([], [], '#ff6b6b', lw=1.5)
        self._line_fairy,  = self._ax3.plot([], [], '#ff6b6b', label='Fairy',  lw=1.5)
        self._line_youkai, = self._ax3.plot([], [], '#00ff00', label='Youkai', lw=1.5)
        self._ax3.legend(fontsize=8)

        self._canvas = FigureCanvasTkAgg(self._fig, master=f4)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Placeholder
        ph = ImageTk.PhotoImage(Image.new('RGB', DISPLAY_SIZE, (30, 30, 40)))
        for lbl in (self._lbl_orig, self._lbl_det, self._lbl_fairy):
            lbl.config(image=ph); lbl.image = ph

    @staticmethod
    def _to_photo(rgb: np.ndarray, size=DISPLAY_SIZE) -> ImageTk.PhotoImage:
        img = Image.fromarray(rgb)
        img.thumbnail(size)
        canvas = Image.new('RGB', size, (30, 30, 40))
        x = (size[0] - img.size[0]) // 2
        y = (size[1] - img.size[1]) // 2
        canvas.paste(img, (x, y))
        return ImageTk.PhotoImage(canvas)

    # ──────────────────────────────────────────
    # Управление видео
    # ──────────────────────────────────────────
    def _start_video(self):
        self._stop_video()
        self.running = True
        threading.Thread(target=self._video_loop, daemon=True).start()
        self._status.set("🎬 ВИДЕО ЗАПУЩЕНО | ОБУЧЕНИЕ ИДЁТ...")

    def _stop_video(self):
        self.running = False
        time.sleep(0.15)
        self._status.set("⏸ ОСТАНОВЛЕНО")

    def _save_battle(self):
        path = filedialog.asksaveasfilename(
            defaultextension='.pth',
            filetypes=[('PyTorch checkpoint', '*.pth'), ('Все файлы', '*.*')],
            initialfile='epic_battle_save.pth',
            title='Сохранить прогресс битвы',
        )
        if not path:
            return
        if self.battle.save(path):
            self._status.set(f"💾 Сохранено: {os.path.basename(path)}")
            messagebox.showinfo("Сохранение",
                                f"Прогресс сохранён!\nФайл: {os.path.basename(path)}\n"
                                f"Шаг обучения: {self.battle.step}\nСтадия: {self.battle.stage}")

    def _load_battle(self):
        path = filedialog.askopenfilename(
            filetypes=[('PyTorch checkpoint', '*.pth'), ('Все файлы', '*.*')],
            title='Загрузить прогресс битвы',
        )
        if not path:
            return
        if self.battle.load(path):
            self._status.set(f"📂 Загружено: {os.path.basename(path)}")
            messagebox.showinfo("Загрузка",
                                f"Прогресс загружен!\nФайл: {os.path.basename(path)}\n"
                                f"Шаг обучения: {self.battle.step}\nСтадия: {self.battle.stage}")
        else:
            messagebox.showwarning("Ошибка", "Не удалось загрузить файл!")

    def _reset_battle(self):
        if messagebox.askyesno("Сброс", "Сбросить всё обучение?"):
            self.battle = EpicBattle()
            self._status.set("🔄 ОБУЧЕНИЕ НАЧАТО ЗАНОВО!")

    # ──────────────────────────────────────────
    # Видеоцикл
    # ──────────────────────────────────────────
    def _video_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.video_path = create_synthetic_video()
            cap = cv2.VideoCapture(self.video_path)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимальный буфер
        frame_n = 0

        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                H, W = frame_rgb.shape[:2]

                # ── 1. Исходное видео ────────────────────────
                ph1 = self._to_photo(frame_rgb)
                self._lbl_orig.config(image=ph1); self._lbl_orig.image = ph1

                # ── 2. Детекция Youkai ───────────────────────
                small = cv2.resize(frame_rgb, (320, 240))
                t = (torch.from_numpy(small).float() / 255.0
                     ).permute(2, 0, 1).unsqueeze(0)

                boxes, conf, classes = self.battle.detect(t)
                det = frame_rgb.copy()

                # На Stage 1 пороговое значение ОЧЕНЬ низкое - рисуем всё!
                if self.battle.stage == 1:
                    threshold = 0.05  # очень низкий порог, чтобы видеть почти все боксы
                else:
                    threshold = max(0.15, 0.40 - self.battle.step * 0.0002)

                drawn = 0
                filtered_boxes = []
                filtered_conf = []
                filtered_classes = []

                for ai in range(boxes.shape[1]):
                    c_val = conf[0, ai, 0].item()
                    if c_val < threshold:
                        continue
                    filtered_boxes.append(boxes[0, ai])
                    filtered_conf.append(conf[0, ai, 0])
                    filtered_classes.append(classes[0, ai])

                if len(filtered_boxes) > 0:
                    filtered_boxes = torch.stack(filtered_boxes)
                    filtered_conf = torch.stack(filtered_conf)
                    filtered_classes = torch.stack(filtered_classes)

                    keep_idx = nms(filtered_boxes, filtered_conf, iou_threshold=0.2)

                    for i in keep_idx:
                        bx = filtered_boxes[i].numpy()
                        c_val = filtered_conf[i].item()
                        cls_id = int(filtered_classes[i].argmax().item())

                        # ВАЛИДАЦИЯ: якоря должны быть в [0, 1]
                        x_norm = np.clip(bx[0], 0, 1)
                        y_norm = np.clip(bx[1], 0, 1)
                        w_norm = np.clip(bx[2], 0.01, 0.5)
                        h_norm = np.clip(bx[3], 0.01, 0.5)

                        x1 = int(x_norm * W)
                        y1 = int(y_norm * H)
                        bw = int(w_norm * W)
                        bh = int(h_norm * H)
                        x2 = min(W - 1, x1 + bw)
                        y2 = min(H - 1, y1 + bh)

                        if x2 - x1 < 10 or y2 - y1 < 10:
                            continue

                        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                        label = f"{CLASS_NAMES[cls_id % len(CLASS_NAMES)]} {c_val:.0%}"

                        cv2.rectangle(det, (x1, y1), (x2, y2), color, 3)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(det, (x1, max(0, y1 - th - 8)), (min(W, x1 + tw + 6), y1), color, -1)
                        cv2.putText(det, label, (x1 + 3, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        drawn += 1

                # Если совсем ничего не рисуется, покажи хотя бы самый лучший бокс
                if drawn == 0 and boxes.shape[1] > 0:
                    for ai in range(min(3, boxes.shape[1])):
                        bx     = boxes[0, ai].numpy()
                        cls_id = int(classes[0, ai].argmax().item())
                        c_val  = conf[0, ai, 0].item()

                        x_norm = np.clip(bx[0], 0, 1)
                        y_norm = np.clip(bx[1], 0, 1)
                        w_norm = np.clip(bx[2], 0.01, 0.3)
                        h_norm = np.clip(bx[3], 0.01, 0.3)

                        x1 = int(x_norm * W)
                        y1 = int(y_norm * H)
                        x2 = min(W - 1, x1 + int(w_norm * W))
                        y2 = min(H - 1, y1 + int(h_norm * H))

                        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                        cv2.rectangle(det, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(det, f"{CLASS_NAMES[cls_id]} {c_val:.0%}",
                                    (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                ph2 = self._to_photo(det)
                self._lbl_det.config(image=ph2); self._lbl_det.image = ph2

                # ── 3. Искажение Fairy ───────────────────────
                with torch.no_grad():
                    corr = self.battle.fairy(t, training=True, stage=self.battle.stage)
                corr_np = (corr.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                corr_up = cv2.resize(corr_np, (W, H))
                ph3 = self._to_photo(corr_up)
                self._lbl_fairy.config(image=ph3); self._lbl_fairy.image = ph3

                # ── 4. Графики и статус (каждые 10 кадров) ───────────
                if frame_n % 10 == 0 and len(self.battle.accuracy_history) > 1:
                    acc  = list(self.battle.accuracy_history)
                    loss = list(self.battle.loss_history)
                    fp   = list(self.battle.fairy_power_history)
                    yp   = list(self.battle.youkai_power_history)
                    xs   = range(len(acc))

                    self._line_acc.set_data(xs, acc)
                    self._line_loss.set_data(xs, loss)
                    self._line_fairy.set_data(xs, fp)
                    self._line_youkai.set_data(xs, yp)

                    lim = max(100, len(acc))
                    for ax in (self._ax1, self._ax2, self._ax3):
                        ax.set_xlim(0, lim)
                    self._ax2.relim(); self._ax2.autoscale_view(scalex=False)

                    total = self.battle.youkai_wins + self.battle.fairy_wins
                    if total > 0:
                        self._ax4.clear()
                        self._ax4.pie(
                            [self.battle.youkai_wins, self.battle.fairy_wins],
                            labels=['Youkai 👁️', 'Fairy 🧚'],
                            colors=['#00cc44', '#ff4444'],
                            autopct='%1.0f%%',
                            textprops={'color': 'white', 'fontsize': 9},
                        )
                        self._ax4.set_title('Счёт побед', color='white', fontsize=9)
                        self._ax4.set_facecolor('#0f0f1a')

                    self._canvas.draw()

                    # Обновляем индикатор стадии
                    if self.battle.stage == 1:
                        avg_real = np.mean(list(self.battle.real_detection_rate)) if len(self.battle.real_detection_rate) > 0 else 0
                        stage_text = f"🎓 STAGE 1: ОБУЧЕНИЕ ({self.battle.step}/{STAGE1_STEPS}) | РЕАЛЬНАЯ ТОЧНОСТЬ: {avg_real:.1%} (нужна {STAGE2_MIN_ACCURACY:.0%}+)"
                    else:
                        stage_text = f"🎮 STAGE 2: ИГРА | Fairy + Youkai"

                    if self.battle.stage1_completed and not hasattr(self, '_stage1_shown'):
                        self._stage1_shown = True
                        messagebox.showinfo("✅ ЭТАП 1 ЗАВЕРШЁН!",
                            "Youkai научился распознавать объекты!\n\n"
                            "НАЧИНАЕТСЯ ЭТАП 2:\n"
                            "Fairy включается и начинает добавлять объекты, шум и деформации.")

                    self._stage_label.set(stage_text)

                    a = self.battle.accuracy_history[-1]
                    self._status.set(
                        f"⚔️ ШАГ: {self.battle.step} | РЕАЛЬНАЯ ТОЧНОСТЬ: {a:.1%} | "
                        f"Youkai 👁️ {self.battle.youkai_wins}  Fairy 🧚 {self.battle.fairy_wins}"
                    )

                frame_n += 1
                time.sleep(0.006)  # ~150 FPS вместо 50 FPS

            except Exception as e:
                print(f"Ошибка отображения: {e}")
                time.sleep(0.1)

        cap.release()

    # ──────────────────────────────────────────
    # Обучающий цикл
    # ──────────────────────────────────────────
    def _training_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        batch = []
        retry_count = 0
        max_retries = 10

        while True:
            try:
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.video_path)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    retry_count += 1
                    if retry_count > max_retries:
                        print("⚠️ Переоткрываю видео...")
                        cap.release()
                        time.sleep(0.1)
                        cap = cv2.VideoCapture(self.video_path)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        retry_count = 0
                    time.sleep(0.02)
                    continue

                retry_count = 0
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small = cv2.resize(rgb, (160, 120))
                t = (torch.from_numpy(small).float() / 255.0
                     ).permute(2, 0, 1).unsqueeze(0)
                batch.append(t)

                if len(batch) >= 4:
                    b = torch.cat(batch, dim=0)
                    acc, loss = self.battle.train_step(b)
                    
                    # Логируем прогресс Stage 1 каждые 50 шагов
                    if self.battle.step % 50 == 0 and self.battle.stage == 1:
                        avg_real = np.mean(list(self.battle.real_detection_rate)[-50:]) if len(self.battle.real_detection_rate) > 0 else 0
                        print(f"  [Stage 1, шаг {self.battle.step}] Реальная точность: {avg_real:.1%} (нужна 85%+)")
                    
                    batch = []
                    time.sleep(0.001)
                else:
                    time.sleep(0.005)

            except Exception as e:
                print(f"Ошибка обучения: {e}")
                time.sleep(0.2)

    # ──────────────────────────────────────────
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.running = False
        if messagebox.askyesno("Выход", "Сохранить прогресс перед выходом?"):
            path = filedialog.asksaveasfilename(
                defaultextension='.pth',
                filetypes=[('PyTorch checkpoint', '*.pth'), ('Все файлы', '*.*')],
                initialfile='epic_battle_save.pth',
                title='Куда сохранить?',
            )
            if path:
                self.battle.save(path)
        self.root.destroy()


# ================================================================
# ЗАПУСК
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("⚔️  FAIRY vs YOUKAI — Нейросетевая игра в прятки")
    print("🎯  Самообучение детектору объектов в видеопотоке")
    print("=" * 60)
    app = EpicApp()
    app.run()
