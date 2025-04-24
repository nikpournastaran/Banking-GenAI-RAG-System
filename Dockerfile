FROM python:3.11-slim

WORKDIR /app

# Копирование файла зависимостей
COPY requirements.txt .

# Создание и активация виртуального окружения
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Обновление pip
RUN pip install --upgrade pip

# Поэтапная установка зависимостей в определенном порядке для избежания конфликтов
# 1. Установка базовых научных библиотек
RUN pip install --no-cache-dir numpy==1.26.1 pandas==2.1.3

# 2. Установка pydantic нужной версии (>= 2.7.4 для совместимости с langchain)
RUN pip install --no-cache-dir pydantic==2.7.4 pydantic-core==2.14.6

# 3. Установка FastAPI и связанных компонентов
RUN pip install --no-cache-dir fastapi==0.104.1 uvicorn==0.23.2 python-dotenv==1.0.0 python-multipart==0.0.6

# 4. Установка OpenAI компонентов
RUN pip install --no-cache-dir openai==1.10.0 tiktoken==0.5.2

# 5. Установка LangChain и связанных компонентов
RUN pip install --no-cache-dir langchain==0.3.0 langchain-community==0.3.0 langchain-openai==0.1.0 langchain-text-splitters==0.3.0 faiss-cpu==1.7.4

# 6. Установка инструментов для работы с документами
RUN pip install --no-cache-dir pypdf==3.17.1 docx2txt==0.8 markdown==3.5 PyPDF2==3.0.1

# 7. Установка HTTP клиента и вспомогательных библиотек
RUN pip install --no-cache-dir httpx==0.25.1 tenacity==8.2.3

# 8. Установка NLTK и python-magic
RUN pip install --no-cache-dir nltk==3.8.1 python-magic==0.4.27

# 9. Установка unstructured в последнюю очередь (т.к. имеет много зависимостей)
RUN pip install --no-cache-dir unstructured==0.10.30

# Установка системных зависимостей для unstructured и git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Копирование остальных файлов проекта
COPY main.py .
COPY static /app/static
COPY index/parts /app/index/parts
COPY start.sh .

# Создание директории для документов, но БЕЗ faiss_index
RUN mkdir -p docs
RUN mkdir -p index
RUN touch last_updated.txt rebuild_log.txt

# Сделать скрипт запуска исполняемым
RUN chmod +x start.sh

# Порт для FastAPI
EXPOSE 8000

# Запуск FastAPI приложения
CMD ["./start.sh"]