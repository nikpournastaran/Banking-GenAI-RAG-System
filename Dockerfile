FROM python:3.11-slim

# Установка рабочей директории
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Обновляем pip и создаём виртуальное окружение
RUN python -m venv /opt/venv \
 && . /opt/venv/bin/activate \
 && pip install --upgrade pip

# Установка системных зависимостей для unstructured и других библиотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python-зависимости
RUN . /opt/venv/bin/activate \
 && pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта
COPY . .

# Делаем стартовый скрипт исполняемым
RUN chmod +x start.sh

# Указываем порт для FastAPI
EXPOSE 8000

# Устанавливаем переменные окружения для виртуального окружения
ENV PATH="/opt/venv/bin:$PATH"

# Запускаем приложение
CMD ["./start.sh"]
