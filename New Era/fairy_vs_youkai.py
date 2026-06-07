#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚔️ FAIRY vs YOUKAI - Эпическая битва нейросетей
Самообучение распознаванию объектов на реальном видео (Shibuya Crossing)
Запуск: python fairy_vs_youkai.py
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
from tkinter import ttk, messagebox
import threading
import time
from collections import deque
import random
import os
import subprocess
import sys
import gc

# Очистка памяти
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Принудительно используем CPU для стабильности
device = torch.device('cpu')
print(f"✅ Используется устройство: {device}")
print(f"✅ PyTorch версия: {torch.__version__}")

# ================================================================
# 1. СКАЧИВАНИЕ ВИДЕО
# ================================================================

def download_shibuya_video():
    """Скачивает видео с пешеходным переходом в Сибуе"""
    video_path = "shibuya_crossing.mp4"
    
    if os.path.exists(video_path):
        size_mb = os.path.getsize(video_path) // (1024 * 1024)
        print(f"✅ Видео уже есть: {video_path} ({size_mb} МБ)")
        return video_path
    
    print("📥 Скачиваю видео с перекрёстком Сибуя (Токио)...")
    print("⏳ Это займёт 1-2 минуты...")
    
    video_url = "https://www.youtube.com/watch?v=4SvwUbDQZmc"
    
    try:
        # Пробуем через yt-dlp
        cmd = f'yt-dlp -f "best[height<=480]" -o "{video_path}" --no-playlist "{video_url}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if os.path.exists(video_path) and os.path.getsize(video_path) > 1000000:
            print(f"✅ Видео успешно скачано!")
            return video_path
    except Exception as e:
        print(f"⚠️ Ошибка скачивания: {e}")
    
    print("⚠️ Создаю синтетическое видео с объектами...")
    return create_synthetic_video()

def create_synthetic_video():
    """Создаёт синтетическое видео с объектами"""
    video_path = "synthetic_training.mp4"
    
    if os.path.exists(video_path):
        return video_path
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 25.0, (640, 480))
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    texts = ['PERSON', 'CAR', 'DOG']
    
    print("🎬 Создаю тренировочное видео...")
    for frame_num in range(300):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Фон
        frame[0:240, :] = [135, 206, 235]
        frame[240:480, :] = [34, 139, 34]
        cv2.rectangle(frame, (0, 350), (640, 430), (80, 80, 80), -1)
        cv2.line(frame, (0, 390), (640, 390), (255, 255, 255), 2)
        
        # Объекты
        for i in range(5):
            color = random.choice(colors)
            text = random.choice(texts)
            x = int((frame_num * 3 + i * 120) % 600)
            y = random.randint(340, 410)
            
            cv2.rectangle(frame, (x, y), (x+50, y+50), color, -1)
            cv2.putText(frame, text, (x+5, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
        if frame_num % 50 == 0:
            print(f"  Прогресс: {frame_num}/300")
    
    out.release()
    print(f"✅ Создано синтетическое видео")
    return video_path

# ================================================================
# 2. НЕЙРОСЕТИ
# ================================================================

class ElegantFairy(nn.Module):
    """Fairy - прятальщик"""
    def __init__(self):
        super(ElegantFairy, self).__init__()
        self.noise_strength = nn.Parameter(torch.tensor(0.12))
        self.warp_strength = nn.Parameter(torch.tensor(0.08))
        
    def forward(self, x, training=True):
        if not training:
            return x
            
        noise_amp = torch.sigmoid(self.noise_strength) * 0.2
        noise = torch.randn_like(x) * noise_amp
        
        warp_amp = torch.sigmoid(self.warp_strength) * 0.1
        batch_size, channels, h, w = x.shape
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing='ij'
        )
        
        warp_x = grid_x + torch.sin(grid_y * 3.14) * warp_amp
        warp_y = grid_y + torch.cos(grid_x * 3.14) * warp_amp
        
        grid = torch.stack([warp_x, warp_y], dim=-1)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        warped = F.grid_sample(x, grid, align_corners=False, padding_mode='border')
        corrupted = warped + noise
        
        return torch.clamp(corrupted, 0, 1)


class ElegantYoukai(nn.Module):
    """Youkai - искатель"""
    def __init__(self, num_classes=5):
        super(ElegantYoukai, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4 + num_classes)
        )
        
    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        
        features = f3.view(f3.size(0), -1)
        output = torch.sigmoid(self.fc(features))
        
        return output[:, :4], output[:, 4:]

# ================================================================
# 3. КЛАСС БИТВЫ
# ================================================================

class EpicBattle:
    def __init__(self):
        self.device = torch.device('cpu')
        print(f"⚔️ Создаю бойцов...")
        
        self.fairy = ElegantFairy().to(self.device)
        self.youkai = ElegantYoukai(num_classes=5).to(self.device)
        
        self.optim_fairy = optim.Adam(self.fairy.parameters(), lr=0.0005)
        self.optim_youkai = optim.Adam(self.youkai.parameters(), lr=0.001)
        
        self.fairy_momentum = 1.0
        self.youkai_momentum = 1.0
        
        self.accuracy_history = deque(maxlen=200)
        self.loss_history = deque(maxlen=200)
        self.fairy_power_history = deque(maxlen=200)
        self.youkai_power_history = deque(maxlen=200)
        self.step = 0
        self.fairy_wins = 0
        self.youkai_wins = 0
        
    def train_step(self, frames):
        frames = frames.to(self.device)
        
        corrupted = self.fairy(frames, training=True)
        bbox_orig, class_orig = self.youkai(frames)
        bbox_corr, class_corr = self.youkai(corrupted)
        
        consistency_loss = F.mse_loss(bbox_orig, bbox_corr) + F.mse_loss(class_orig, class_corr)
        accuracy = 1.0 / (1.0 + consistency_loss.item())
        
        # Динамический баланс
        if accuracy > 0.6:
            self.youkai_wins += 1
            self.fairy_momentum = min(2.0, self.fairy_momentum * 1.02)
        elif accuracy < 0.4:
            self.fairy_wins += 1
            self.youkai_momentum = min(2.0, self.youkai_momentum * 1.02)
        else:
            self.fairy_momentum = max(0.5, self.fairy_momentum * 0.99)
            self.youkai_momentum = max(0.5, self.youkai_momentum * 0.99)
        
        youkai_loss = consistency_loss * self.youkai_momentum
        fairy_loss = -consistency_loss * self.fairy_momentum + 0.01 * torch.mean(torch.abs(corrupted - frames))
        
        self.optim_youkai.zero_grad()
        youkai_loss.backward(retain_graph=True)
        self.optim_youkai.step()
        
        self.optim_fairy.zero_grad()
        fairy_loss.backward()
        self.optim_fairy.step()
        
        self.accuracy_history.append(accuracy)
        self.loss_history.append(consistency_loss.item())
        self.fairy_power_history.append(self.fairy_momentum)
        self.youkai_power_history.append(self.youkai_momentum)
        self.step += 1
        
        return accuracy, consistency_loss.item()
    
    def detect(self, frame):
        with torch.no_grad():
            bbox, classes = self.youkai(frame.to(self.device))
        return bbox.cpu(), classes.cpu()
    
    def save(self, path='epic_battle_save.pth'):
        torch.save({
            'fairy': self.fairy.state_dict(),
            'youkai': self.youkai.state_dict(),
            'step': self.step,
            'fairy_wins': self.fairy_wins,
            'youkai_wins': self.youkai_wins,
            'fairy_momentum': self.fairy_momentum,
            'youkai_momentum': self.youkai_momentum,
        }, path)
        return True
    
    def load(self, path='epic_battle_save.pth'):
        if os.path.exists(path):
            data = torch.load(path, map_location=self.device)
            self.fairy.load_state_dict(data['fairy'])
            self.youkai.load_state_dict(data['youkai'])
            self.step = data['step']
            self.fairy_wins = data['fairy_wins']
            self.youkai_wins = data['youkai_wins']
            self.fairy_momentum = data['fairy_momentum']
            self.youkai_momentum = data['youkai_momentum']
            return True
        return False

# ================================================================
# 4. ГЛАВНОЕ ПРИЛОЖЕНИЕ
# ================================================================

class EpicApp:
    def __init__(self):
        self.root = tk.Tk()
    
        self.root.geometry("1600x950")
        self.root.configure(bg='#1a1a2e')
        
        # Скачиваем видео
        print("\n📹 Подготовка видео...")
        self.video_path = download_shibuya_video()
        
        # Создаём битву
        print("🧠 Инициализация нейросетей...")
        self.battle = EpicBattle()
        if self.battle.load():
            print(f"✅ Загружен прогресс: шаг {self.battle.step}")
        
        self.cap = None
        self.running = False
        
        self.class_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        self.class_names = ['ЧЕЛОВЕК', 'МАШИНА', 'СОБАКА', 'АВТОБУС', 'ВЕЛОСИПЕД']
        
        self.setup_ui()
        self.start_video()
        
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()
        
        print("\n✅ Приложение запущено!")
        print("=" * 50)
        
    def setup_ui(self):
        # Заголовок
        title_frame = tk.Frame(self.root, bg='#1a1a2e', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title = tk.Label(title_frame, text="⚔️ FAIRY vs YOUKAI ⚔️", 
                        font=('Arial', 24, 'bold'), bg='#1a1a2e', fg='#e94560')
        title.pack(pady=15)
        
        subtitle = tk.Label(title_frame, text="Самообучение распознаванию объектов | Перекрёсток Сибуя, Токио", 
                           font=('Arial', 10), bg='#1a1a2e', fg='#888')
        subtitle.pack()
        
        # Панель управления
        control_frame = tk.Frame(self.root, bg='#0f0f1a', height=50)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        control_frame.pack_propagate(False)
        
        btn_style = {'bg': '#e94560', 'fg': 'white', 'font': ('Arial', 10, 'bold'), 
                    'relief': 'raised', 'bd': 2, 'padx': 15, 'pady': 5}
        
        tk.Button(control_frame, text="▶ СТАРТ", command=self.start_video, **btn_style).pack(side=tk.LEFT, padx=5, pady=8)
        tk.Button(control_frame, text="⏹ СТОП", command=self.stop_video, **btn_style).pack(side=tk.LEFT, padx=5, pady=8)
        tk.Button(control_frame, text="💾 СОХРАНИТЬ", command=self.save_battle, **btn_style).pack(side=tk.LEFT, padx=5, pady=8)
        tk.Button(control_frame, text="📂 ЗАГРУЗИТЬ", command=self.load_battle, **btn_style).pack(side=tk.LEFT, padx=5, pady=8)
        tk.Button(control_frame, text="🔄 СБРОС", command=self.reset_battle, **btn_style).pack(side=tk.LEFT, padx=5, pady=8)
        
        self.status_var = tk.StringVar(value="⚔️ БИТВА ГОТОВА К НАЧАЛУ")
        status_label = tk.Label(control_frame, textvariable=self.status_var, bg='#0f0f1a', 
                               fg='#00ff00', font=('Arial', 11, 'bold'))
        status_label.pack(side=tk.RIGHT, padx=20)
        
        # Основная область
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Верхний ряд
        top_frame = tk.Frame(main_frame, bg='#1a1a2e')
        top_frame.pack(fill=tk.BOTH, expand=True)
        
        frame1 = tk.LabelFrame(top_frame, text="📹 1. ИСХОДНОЕ ВИДЕО (Перекрёсток Сибуя)", 
                              bg='#16213e', fg='white', font=('Arial', 11, 'bold'))
        frame1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.label_original = tk.Label(frame1, bg='black')
        self.label_original.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        frame2 = tk.LabelFrame(top_frame, text="👁️ 2. YOUKAI - ОБНАРУЖЕНИЕ ОБЪЕКТОВ", 
                              bg='#16213e', fg='#00ff00', font=('Arial', 11, 'bold'))
        frame2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.label_detection = tk.Label(frame2, bg='black')
        self.label_detection.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Нижний ряд
        bottom_frame = tk.Frame(main_frame, bg='#1a1a2e')
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        frame3 = tk.LabelFrame(bottom_frame, text="🧚 3. FAIRY - ИСКАЖЕНИЕ ВИДЕО", 
                              bg='#16213e', fg='#ff6b6b', font=('Arial', 11, 'bold'))
        frame3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.label_fairy = tk.Label(frame3, bg='black')
        self.label_fairy.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        frame4 = tk.LabelFrame(bottom_frame, text="📊 4. СТАТИСТИКА БИТВЫ", 
                              bg='#16213e', fg='white', font=('Arial', 11, 'bold'))
        frame4.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Графики
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(8, 6))
        self.fig.patch.set_facecolor('#16213e')
        
        self.ax1.set_title('Точность Youkai', color='white', fontsize=10)
        self.ax1.set_facecolor('#0f0f1a')
        self.ax1.tick_params(colors='white')
        self.ax1.set_ylim(0, 1)
        self.ax1.grid(True, alpha=0.3)
        self.line_acc, = self.ax1.plot([], [], '#00ff00', linewidth=2)
        
        self.ax2.set_title('Функция потерь', color='white', fontsize=10)
        self.ax2.set_facecolor('#0f0f1a')
        self.ax2.tick_params(colors='white')
        self.ax2.grid(True, alpha=0.3)
        self.line_loss, = self.ax2.plot([], [], '#ff6b6b', linewidth=2)
        
        self.ax3.set_title('Сила противников', color='white', fontsize=10)
        self.ax3.set_facecolor('#0f0f1a')
        self.ax3.tick_params(colors='white')
        self.ax3.set_ylim(0, 2)
        self.ax3.grid(True, alpha=0.3)
        self.fairy_line, = self.ax3.plot([], [], '#ff6b6b', label='Fairy', linewidth=2)
        self.youkai_line, = self.ax3.plot([], [], '#00ff00', label='Youkai', linewidth=2)
        self.ax3.legend(loc='upper right')
        
        self.ax4.set_title('Счёт побед', color='white', fontsize=10)
        self.ax4.set_facecolor('#0f0f1a')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame4)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Placeholder
        placeholder = Image.new('RGB', (640, 480), (30, 30, 40))
        photo = ImageTk.PhotoImage(placeholder)
        self.label_original.config(image=photo)
        self.label_original.image = photo
        self.label_detection.config(image=photo)
        self.label_detection.image = photo
        self.label_fairy.config(image=photo)
        self.label_fairy.image = photo
        
    def start_video(self):
        self.stop_video()
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.video_path = create_synthetic_video()
            self.cap = cv2.VideoCapture(self.video_path)
        
        self.running = True
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        self.status_var.set("🎬 ВИДЕО ЗАПУЩЕНО | ОБУЧЕНИЕ ИДЁТ...")
        
    def stop_video(self):
        self.running = False
        time.sleep(0.1)
        self.status_var.set("⏸ ВИДЕО ОСТАНОВЛЕНО")
        
    def save_battle(self):
        self.battle.save()
        self.status_var.set("💾 БИТВА СОХРАНЕНА!")
        messagebox.showinfo("Сохранение", "Прогресс битвы сохранён!")
        
    def load_battle(self):
        if self.battle.load():
            self.status_var.set("📂 БИТВА ЗАГРУЖЕНА!")
            messagebox.showinfo("Загрузка", "Прогресс битвы загружен!")
        else:
            messagebox.showwarning("Ошибка", "Нет сохранённой битвы!")
            
    def reset_battle(self):
        if messagebox.askyesno("Сброс", "Сбросить всё обучение?"):
            self.battle = EpicBattle()
            self.status_var.set("🔄 БИТВА НАЧАТА ЗАНОВО!")
            
    def video_loop(self):
        frame_count = 0
        
        while self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
                
                # Исходное видео
                img = Image.fromarray(frame_rgb)
                img.thumbnail((640, 480))
                final = Image.new('RGB', (640, 480), (30, 30, 40))
                x = (640 - img.size[0]) // 2
                y = (480 - img.size[1]) // 2
                final.paste(img, (x, y))
                photo = ImageTk.PhotoImage(final)
                self.label_original.config(image=photo)
                self.label_original.image = photo
                
                # Детекция Youkai
                with torch.no_grad():
                    bbox, classes = self.battle.detect(frame_tensor)
                
                det_frame = frame_rgb.copy()
                h, w = det_frame.shape[:2]
                bbox_np = bbox.numpy()[0]
                
                if len(bbox_np) >= 4:
                    x = int(max(0, min(w, bbox_np[0] * w)))
                    y = int(max(0, min(h, bbox_np[1] * h)))
                    w_box = int(max(10, min(w - x, bbox_np[2] * w)))
                    h_box = int(max(10, min(h - y, bbox_np[3] * h)))
                    
                    class_id = 0
                    if len(classes.numpy()[0]) > 0:
                        class_id = np.argmax(classes.numpy()[0])
                    
                    color = self.class_colors[class_id % len(self.class_colors)]
                    cv2.rectangle(det_frame, (x, y), (x+w_box, y+h_box), color, 3)
                    label = f"{self.class_names[class_id % len(self.class_names)]}"
                    cv2.putText(det_frame, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                img2 = Image.fromarray(det_frame)
                img2.thumbnail((640, 480))
                final2 = Image.new('RGB', (640, 480), (30, 30, 40))
                x2 = (640 - img2.size[0]) // 2
                y2 = (480 - img2.size[1]) // 2
                final2.paste(img2, (x2, y2))
                photo2 = ImageTk.PhotoImage(final2)
                self.label_detection.config(image=photo2)
                self.label_detection.image = photo2
                
                # Искажение Fairy
                with torch.no_grad():
                    corrupted = self.battle.fairy(frame_tensor, training=False)
                corr_np = corrupted.squeeze().permute(1, 2, 0).numpy()
                corr_np = (corr_np * 255).astype(np.uint8)
                
                img3 = Image.fromarray(corr_np)
                img3.thumbnail((640, 480))
                final3 = Image.new('RGB', (640, 480), (30, 30, 40))
                x3 = (640 - img3.size[0]) // 2
                y3 = (480 - img3.size[1]) // 2
                final3.paste(img3, (x3, y3))
                photo3 = ImageTk.PhotoImage(final3)
                self.label_fairy.config(image=photo3)
                self.label_fairy.image = photo3
                
                # Графики
                if frame_count % 10 == 0 and len(self.battle.accuracy_history) > 0:
                    acc_data = list(self.battle.accuracy_history)
                    loss_data = list(self.battle.loss_history)
                    fairy_power = list(self.battle.fairy_power_history)
                    youkai_power = list(self.battle.youkai_power_history)
                    
                    x_plot = range(len(acc_data))
                    self.line_acc.set_data(x_plot, acc_data)
                    self.line_loss.set_data(x_plot, loss_data)
                    self.fairy_line.set_data(x_plot, fairy_power)
                    self.youkai_line.set_data(x_plot, youkai_power)
                    
                    self.ax1.set_xlim(0, max(100, len(acc_data)))
                    self.ax2.set_xlim(0, max(100, len(loss_data)))
                    self.ax3.set_xlim(0, max(100, len(acc_data)))
                    
                    total = self.battle.youkai_wins + self.battle.fairy_wins
                    if total > 0:
                        self.ax4.clear()
                        self.ax4.pie([self.battle.youkai_wins, self.battle.fairy_wins], 
                                    labels=['Youkai', 'Fairy'],
                                    colors=['#00ff00', '#ff6b6b'],
                                    autopct='%1.0f%%')
                        self.ax4.set_title('Счёт побед', color='white', fontsize=10)
                        self.ax4.set_facecolor('#0f0f1a')
                    
                    self.canvas.draw()
                    
                    acc = self.battle.accuracy_history[-1]
                    self.status_var.set(f"⚔️ ШАГ: {self.battle.step} | ТОЧНОСТЬ: {acc:.1%}")
                
                frame_count += 1
                time.sleep(0.04)
                
            except Exception as e:
                print(f"Ошибка: {e}")
                time.sleep(0.1)
    
    def training_loop(self):
        batch_frames = []
        
        while True:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue
                else:
                    time.sleep(0.05)
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
                
                batch_frames.append(frame_tensor)
                
                if len(batch_frames) >= 4:
                    batch = torch.cat(batch_frames, dim=0)
                    self.battle.train_step(batch)
                    batch_frames = []
                    time.sleep(0.02)
                else:
                    time.sleep(0.05)
                    
            except Exception as e:
                print(f"Ошибка обучения: {e}")
                time.sleep(0.1)
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
        
    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.battle.save()
        self.root.destroy()


# ================================================================
# 5. ЗАПУСК
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("⚔️ ЗАПУСК ЭПИЧЕСКОЙ БИТВЫ FAIRY vs YOUKAI ⚔️")
    print("📍 Локация: Перекрёсток Сибуя, Токио")
    print("🎯 Цель: Самообучение распознаванию объектов")
    print("=" * 60)
    
    app = EpicApp()
    app.run()