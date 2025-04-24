# Используем минимальный образ Python
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Создаем виртуальное окружение и активируем
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Устанавливаем системные зависимости (до pip install, чтобы кэш не сбрасывался)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN pip install --upgrade pip

# Устанавливаем все зависимости одним шагом — это лучше для кэша Docker
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта
COPY main.py .
COPY static /app/static
COPY index/parts /app/index/parts
COPY start.sh .

# Создаем необходимые директории и файлы
RUN mkdir -p docs index && \
    touch last_updated.txt rebuild_log.txt && \
    chmod +x start.sh

# Открываем порт приложения
EXPOSE 8000

# Команда запуска
CMD ["./start.sh"]
