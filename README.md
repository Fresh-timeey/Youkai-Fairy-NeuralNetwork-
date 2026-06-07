<div align="center">

# 🧿 FAIRY + YOUKAI
### Самообучающаяся система распознавания объектов на основе состязательного обучения

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFAA?style=for-the-badge)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> *Система запускается на любой камере, в любом месте — и через несколько минут знает это место лучше, чем модель, обученная на миллионах чужих картинок.*

</div>

---

## 📌 Постановка задачи

Современные нейросети блестяще распознают объекты — **на тех данных, на которых их обучили**. При переносе в новую среду (склад, улица, производственный цех) начинаются ошибки: другой свет, другие углы, другие объекты. Классическая адаптация требует тысяч вручную размеченных изображений и недель работы.

**Вопрос:** можно ли создать систему, которая адаптируется сама — прямо на месте, в реальном времени, без единой вручную размеченной картинки?

**Ответ:** да. Это FAIRY + YOUKAI.

---

## 🧠 Архитектура системы

```
┌─────────────────────────────────────────────────────────┐
│                        VIDEO STREAM                      │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │       FairyYOLO         │  ← YOLOv8n (заморожен backbone)
          │   Детектор-учитель      │     Генерирует псевдометки
          └────────────┬────────────┘
                       │ pseudo_labels (conf ≥ 0.6)
          ┌────────────▼────────────┐
          │     FairyDistorter      │  ← Вносит аномалии (BLACKOUT / DUPLICATE)
          └────────────┬────────────┘
                       │ distorted frame
          ┌────────────▼────────────┐
          │         YOUKAI          │  ← Кастомная сеть, обучается в реальном времени
          │    Нейросеть-ученик     │     4-блочный backbone + multi-scale neck
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │      AnomalyEngine      │  ← IoU-сопоставление предсказаний двух сетей
          └─────────────────────────┘
```

### Шесть компонентов системы

| Компонент | Роль |
|---|---|
| **FairyYOLO** | Детектор-учитель на базе YOLOv8n, генерирует псевдометки |
| **FairyDistorter** | Вносит искажения в видеопоток для тестирования |
| **YOUKAI** | Кастомная нейросеть оригинальной архитектуры |
| **YoukaiTrainer** | Управление циклом онлайн-обучения YOUKAI на каждом кадре |
| **AnomalyEngine** | Движок сравнения предсказаний обоих детекторов |
| **App** | Главное приложение с 5-панельным интерфейсом (3 потока) |

---

## ⚙️ Как работает обучение

На каждом кадре происходит следующее:

1. **FAIRY** обрабатывает оригинальный кадр → генерирует псевдометки `conf ≥ 0.6`
2. **FairyDistorter** вносит искажение → YOUKAI получает «испорченный» кадр
3. **YOUKAI** делает предсказание на искажённом кадре
4. Вычисляется **функция потерь**:

$$\mathcal{L} = \lambda_1 \cdot L_{cls} + \lambda_2 \cdot L_{loc}$$

где $L_{loc} = 1 - IoU(b_{pred},\, b_{pseudo}) = 1 - \frac{|A \cap B|}{|A \cup B|}$

5. Градиент распространяется назад, параметры обновляются через **Adam**
6. Новые примеры попадают в **Replay Buffer** — стабилизирует обучение

---

## 🎯 Ключевые модификации FAIRY

```python
# Заморозка backbone — первые 10 слоёв не трогаем
for param in model.model[:10].parameters():
    param.requires_grad = False

# Фокус исключительно на классе "person"
results = model(frame, conf=0.6, iou=0.45, classes=[0])
```

---

## 📊 Результаты

| Метрика | Значение |
|---|---|
| Порог активации anomaly engine | `precision ≥ 0.42` |
| Время до конвергенции | **7–12 минут** (4000–7000 шагов) |
| Recall для аномалий BLACKOUT | **80–92%** |
| Аппаратура тестирования | AMD Ryzen 7 260, CPU only |
| Тестовое видео | Перекрёсток Сибуя, Токио (480p, 30 fps) |

> 💡 Ни одного гигабайта предзагруженного датасета. Ни одного вручную размеченного изображения.

---

## 🖼️ Галерея

<div align="center">

### Анатомия системы

| | | |
|:---:|:---:|:---:|
| ![Anatomy](Anatomy.png) | ![Anatomy2](Anatomy2.png) | ![Anatomy3](Anatomy3.png) |
| *Общая архитектура* | *Потоки данных* | *Компоненты детекции* |

---

### Презентация проекта

| | |
|:---:|:---:|
| ![Slide 1](Presentation_1.png) | ![Slide 2](Presentation_2.png) |
| *Постановка задачи* | *Теоретическая база* |

| | |
|:---:|:---:|
| ![Slide 3](Presentation_3.png) | ![Slide 4](Presentation_4.png) |
| *Архитектура FAIRY+YOUKAI* | *Обучение в реальном времени* |

| | |
|:---:|:---:|
| ![Slide 5](Presentation_5.png) | ![Slide 6](Presentation_6.png) |
| *Графический интерфейс* | *AnomalyEngine* |

| | |
|:---:|:---:|
| ![Slide 7](Presentation_7.png) | ![Slide 8](Presentation_8.png) |
| *Результаты апробации* | *Выводы и перспективы* |

</div>

---

## 🌐 Области применения

- 🏭 Промышленный контроль качества
- 🔒 Охрана периметра
- 🚦 Мониторинг транспортных потоков
- 📹 Интеллектуальное видеонаблюдение

---

## 📚 Использованные материалы

<details>
<summary>Список литературы</summary>

- He, K. et al. *Deep Residual Learning for Image Recognition* // CVPR. — 2016.
- Liu, W. et al. *SSD: Single Shot MultiBox Detector* // ECCV. — 2016.
- Redmon, J. et al. *You Only Look Once: Unified, Real-Time Object Detection* // CVPR. — 2016.
- Jocher, G. *Ultralytics YOLOv8*. — 2023. — [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Hinton, G. *Distilling the Knowledge in a Neural Network* // arXiv:1503.02531. — 2015.
- Goodfellow, I. *Deep Learning*. — MIT Press, 2016.
- XI Международная научная конференция по информатике CSIST-2025, Минск.

</details>

---

<div align="center">

**Автор:** Апенко Константин Сергеевич  
**Научный руководитель:** Лобач Сергей Викторович, старший преподаватель кафедры ММАД

<br>

<sub>Использованные технологии и ресурсы: Python · PyTorch · Ultralytics YOLOv8 · OpenCV · Tkinter · Matplotlib · Google Search · материалы MIT OpenCourseWare · DeepLearning.AI (Andrew Ng, Stanford University)</sub>

</div>
