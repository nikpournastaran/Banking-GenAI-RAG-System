from fastapi import FastAPI, Form, Request, Cookie, Response, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import os
import uuid
import html
import time
import hashlib
import subprocess
import tempfile
import json
import shutil
import sys
import traceback
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # ChatOpenAI можно оставить как запасной вариант
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

# Создаем приложение FastAPI
app = FastAPI(
    title="RAG Chat Bot",
    description="Чат-бот с использованием Retrieval-Augmented Generation",
    version="1.0.1"
)

# Настройка CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://standardbusiness.online",
    "https://*.standardbusiness.online",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Проверка директории static перед монтированием
static_dir = "."
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Константы - сохраняем пути к индексу как в новом коде
INDEX_PATH = "/data"  # Основной диск на Render
LOCAL_INDEX_PATH = "./index"  # Локальный путь к индексу в проекте

# Хранение сессий
session_memories = {}
session_last_activity = {}
SESSION_MAX_AGE = 86400  # 24 часа


# Функция для очистки диска Render при необходимости
def clear_render_storage(except_files=None):
    """Очищает директорию индекса на Render, сохраняя указанные файлы"""
    if except_files is None:
        except_files = []  # Файлы, которые не нужно удалять

    print(f"Очистка директории {INDEX_PATH}...")
    try:
        if not os.path.exists(INDEX_PATH):
            print(f"Директория {INDEX_PATH} не существует")
            return True

        # Получаем список всех файлов и директорий
        items = os.listdir(INDEX_PATH)

        for item in items:
            # Пропускаем файлы из исключений
            if item in except_files:
                print(f"Пропуск файла {item} (в списке исключений)")
                continue

            path = os.path.join(INDEX_PATH, item)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Удалена директория: {item}")
                else:
                    os.remove(path)
                    print(f"Удален файл: {item}")
            except Exception as e:
                print(f"Ошибка при удалении {path}: {e}")

        print(f"Директория {INDEX_PATH} успешно очищена")
        return True
    except Exception as e:
        print(f"Ошибка при очистке директории {INDEX_PATH}: {e}")
        return False


# Копирование индекса из локальной директории проекта на Render
def copy_index_to_render_storage(clear_first=True):
    """Копирует индекс из локального проекта в persistent storage на Render"""
    print(f"Копирование индекса из {LOCAL_INDEX_PATH} в {INDEX_PATH}...")

    # Проверяем существует ли локальный индекс
    if not os.path.exists(os.path.join(LOCAL_INDEX_PATH, "index.faiss")):
        print("Ошибка: Локальный индекс не найден")
        return False

    try:
        # Создаем директорию на Render, если её нет
        os.makedirs(INDEX_PATH, exist_ok=True)

        # Очищаем директорию перед копированием, если запрошено
        if clear_first:
            clear_render_storage(except_files=["error.log"])

        # Создаем файл блокировки для предотвращения конфликтов
        lock_file = os.path.join(INDEX_PATH, "index_building.lock")
        with open(lock_file, 'w') as f:
            f.write(f"Index copy started at {datetime.now().isoformat()}")

        # Копируем все файлы из локального индекса в Render storage
        for item in os.listdir(LOCAL_INDEX_PATH):
            source = os.path.join(LOCAL_INDEX_PATH, item)
            destination = os.path.join(INDEX_PATH, item)

            if os.path.isfile(source):
                shutil.copy2(source, destination)
                print(f"Скопирован файл: {item}")
            elif os.path.isdir(source):
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
                print(f"Скопирована директория: {item}")

        # Обновляем информацию о времени копирования
        with open(os.path.join(INDEX_PATH, "copied_at.txt"), "w", encoding="utf-8") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Создаем флаг файл, который показывает, что индекс был скопирован
        with open(os.path.join(INDEX_PATH, "index_copied_flag.txt"), "w", encoding="utf-8") as f:
            f.write("Индекс успешно скопирован: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Удаляем файл блокировки
        if os.path.exists(lock_file):
            os.remove(lock_file)

        print("Индекс успешно скопирован в persistent storage на Render")
        return True

    except Exception as e:
        print(f"Ошибка при копировании индекса: {str(e)}")
        traceback.print_exc()
        return False


# Загружаем векторное хранилище
def load_vectorstore():
    print("Загрузка векторного хранилища...")
    index_file = os.path.join(INDEX_PATH, "index.faiss")

    if not os.path.exists(index_file):
        print("Индекс не найден в persistent storage.")

        # Проверяем наличие локального индекса и копируем его, если есть
        local_index_file = os.path.join(LOCAL_INDEX_PATH, "index.faiss")
        if os.path.exists(local_index_file):
            print("Найден локальный индекс. Копирование в persistent storage...")
            if not copy_index_to_render_storage():
                raise RuntimeError("Не удалось скопировать локальный индекс в persistent storage.")
        else:
            raise RuntimeError("Индекс не найден ни в persistent storage, ни в локальной директории.")

    try:
        print("Попытка загрузки индекса из:", INDEX_PATH)
        vectorstore = FAISS.load_local(INDEX_PATH, OpenAIEmbeddings(model="text-embedding-3-small"),
                                       allow_dangerous_deserialization=True)
        print("Векторное хранилище успешно загружено")

        # Дополнительная проверка векторного хранилища
        try:
            # Простой тестовый запрос для проверки функциональности
            test_query = "тестовый запрос"
            _ = vectorstore.similarity_search(test_query, k=1)
            print("Проверка функциональности индекса успешна")
        except Exception as e:
            print(f"Предупреждение: Индекс загружен, но при тестировании возникла ошибка: {e}")
            # Продолжаем работу, так как основная загрузка прошла успешно

        return vectorstore
    except Exception as e:
        print("Ошибка при загрузке индекса:", e)
        traceback.print_exc()
        raise RuntimeError(f"Индекс найден, но не удалось загрузить: {str(e)}")


# Очистка старых сессий
def clean_old_sessions():
    """Очищает старые сессии для экономии памяти"""
    current_time = time.time()
    expired_sessions = [
        session_id for session_id, last_active in session_last_activity.items()
        if current_time - last_active > SESSION_MAX_AGE
    ]

    for session_id in expired_sessions:
        if session_id in session_memories:
            del session_memories[session_id]
        if session_id in session_last_activity:
            del session_last_activity[session_id]


# Вспомогательная функция для проверки доступности директории
def check_directory_access(directory):
    """Проверяет доступ к директории и выводит информацию о ней"""
    try:
        if os.path.exists(directory):
            print(f"Директория {directory} существует")
            # Проверяем права
            readable = os.access(directory, os.R_OK)
            writable = os.access(directory, os.W_OK)
            executable = os.access(directory, os.X_OK)
            print(f"  Права доступа: Чтение={readable}, Запись={writable}, Выполнение={executable}")

            # Проверяем тип
            is_dir = os.path.isdir(directory)
            is_link = os.path.islink(directory)
            print(f"  Тип: Директория={is_dir}, Символическая ссылка={is_link}")

            # Проверяем содержимое
            if is_dir and readable and executable:
                try:
                    items = os.listdir(directory)
                    print(f"  Содержимое ({len(items)} элементов): {', '.join(items[:5])}" +
                          ("..." if len(items) > 5 else ""))
                except Exception as e:
                    print(f"  Ошибка при чтении содержимого: {str(e)}")
        else:
            parent_dir = os.path.dirname(directory)
            print(f"Директория {directory} не существует")
            print(f"Родительская директория {parent_dir} " +
                  ("существует" if os.path.exists(parent_dir) else "не существует"))

            # Пытаемся создать директорию
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"  Создана директория {directory}")
            except Exception as e:
                print(f"  Не удалось создать директорию: {str(e)}")

        return True
    except Exception as e:
        print(f"Ошибка при проверке директории {directory}: {str(e)}")
        return False


# События приложения
@app.on_event("startup")
async def startup_event():
    print("Запуск приложения...")
    print(f"Текущая рабочая директория: {os.getcwd()}")

    # Проверяем параметры системы
    print(f"Платформа: {sys.platform}")
    print(f"Версия Python: {sys.version}")
    print("Переменные окружения:")
    for env_var in ['RENDER', 'PATH', 'HOME']:
        print(f"  {env_var}={os.environ.get(env_var, 'Не задано')}")

    # Проверка доступности директорий
    print("\nПроверка директорий:")
    check_directory_access(INDEX_PATH)
    check_directory_access(LOCAL_INDEX_PATH)

    # Проверяем наличие индекса в persistent storage
    index_in_persistent = os.path.exists(os.path.join(INDEX_PATH, "index.faiss"))
    print(f"\nИндекс в persistent storage: {'Найден' if index_in_persistent else 'Не найден'}")

    # Проверяем наличие необходимых API ключей
    if not os.environ.get("OPENAI_API_KEY"):
        print("ВНИМАНИЕ: Отсутствует ключ OPENAI_API_KEY для работы эмбеддингов")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ВНИМАНИЕ: Отсутствует ключ ANTHROPIC_API_KEY для работы с Claude")
        print("Бот может работать некорректно без этих ключей.")

    # Проверяем наличие локального индекса
    local_index_file = os.path.join(LOCAL_INDEX_PATH, "index.faiss")
    local_index_exists = os.path.exists(local_index_file)
    print(f"Локальный индекс: {'Найден' if local_index_exists else 'Не найден'}")

    # Стратегия копирования:
    # 1. Если индекса нет в persistent storage, но есть локально - копируем
    # 2. Если индекс есть в persistent storage, но есть более новый локальный - копируем
    # 3. Иначе используем существующий в persistent storage

    if not index_in_persistent and local_index_exists:
        print("Индекс отсутствует в persistent storage. Копирование локального индекса...")
        copy_index_to_render_storage(clear_first=True)
    elif index_in_persistent and local_index_exists:
        # Проверяем даты изменения индексов
        local_mtime = os.path.getmtime(local_index_file)
        persistent_mtime = os.path.getmtime(os.path.join(INDEX_PATH, "index.faiss"))

        if local_mtime > persistent_mtime:
            print("Локальный индекс новее. Обновление индекса в persistent storage...")
            copy_index_to_render_storage(clear_first=True)
        else:
            print("Индекс в persistent storage актуален. Копирование не требуется.")
    elif index_in_persistent:
        print("Используем существующий индекс в persistent storage.")
    else:
        print("ВНИМАНИЕ: Индекс не найден ни в persistent storage, ни локально!")
        print("Приложение может работать некорректно без индекса.")

    print("Приложение запущено и готово к работе!")


# Эндпоинты
@app.get("/ping")
def ping():
    """Проверка работы сервера"""
    # Добавляем информацию о состоянии индекса
    index_exists = os.path.exists(os.path.join(INDEX_PATH, "index.faiss"))
    return {
        "status": "ok",
        "message": "Сервер работает",
        "index_status": "Индекс найден" if index_exists else "Индекс не найден"
    }


@app.post("/update-index")
async def update_index(admin_token: str = Header(None)):
    """Копирует индекс из локальной директории проекта в persistent storage на Render"""
    # Проверка пароля администратора
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_password:
        return JSONResponse({
            "status": "error",
            "message": "Пароль администратора не задан в конфигурации сервера"
        }, status_code=500)

    expected_token = hashlib.sha256(admin_password.encode()).hexdigest()
    if not admin_token or admin_token != expected_token:
        return JSONResponse({
            "status": "error",
            "message": "Доступ запрещен: неверный пароль администратора"
        }, status_code=403)

    # Копирование индекса
    try:
        print("Запрос на копирование индекса из локального проекта в persistent storage...")
        success = copy_index_to_render_storage(clear_first=True)

        if success:
            return JSONResponse({
                "status": "success",
                "message": "Индекс успешно скопирован в persistent storage"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "Не удалось скопировать индекс в persistent storage"
            }, status_code=500)
    except Exception as e:
        error_msg = f"Ошибка при копировании индекса: {str(e)}"
        print(error_msg)
        return JSONResponse({
            "status": "error",
            "message": error_msg
        }, status_code=500)


@app.post("/clear-session")
def clear_session(session_id: str = Cookie(None), response: Response = None):
    """Очищает историю сессии"""
    if session_id and session_id in session_memories:
        session_memories[session_id] = []
        return {"status": "success", "message": "История диалога очищена"}
    else:
        return {"status": "error", "message": "Сессия не найдена"}


@app.get("/index-info")
def get_index_info():
    """Возвращает информацию об индексе"""
    try:
        result = {
            "status": "success",
            "index_location": INDEX_PATH,
            "index_exists": os.path.exists(os.path.join(INDEX_PATH, "index.faiss")),
            "local_index_exists": os.path.exists(os.path.join(LOCAL_INDEX_PATH, "index.faiss")),
        }

        # Добавляем информацию о метаданных, если они есть
        metadata_path = os.path.join(INDEX_PATH, "index_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    result["metadata"] = metadata
            except Exception as e:
                result["metadata_error"] = str(e)

        # Проверяем наличие файла с датой копирования
        copied_at_path = os.path.join(INDEX_PATH, "copied_at.txt")
        if os.path.exists(copied_at_path):
            try:
                with open(copied_at_path, 'r', encoding='utf-8') as f:
                    result["copied_at"] = f.read().strip()
            except Exception as e:
                result["copied_at_error"] = str(e)

        # Добавляем количество документов и чанков в chunk_store, если он есть
        chunk_store_path = os.path.join(INDEX_PATH, "chunk_store.json")
        if os.path.exists(chunk_store_path):
            try:
                with open(chunk_store_path, 'r', encoding='utf-8') as f:
                    chunk_store = json.load(f)
                    result["chunk_count"] = len(chunk_store)
            except Exception as e:
                result["chunk_store_error"] = str(e)

        return result
    except Exception as e:
        return {"status": "error", "message": f"Ошибка при получении информации об индексе: {str(e)}"}


@app.post("/ask")
async def ask(q: str = Form(...), session_id: str = Cookie(None), response: Response = None):
    """Основной эндпоинт для вопросов к чат-боту"""
    print(f"Получен запрос: {q[:50]}...")

    # Проверяем, есть ли текст в запросе
    if not q or len(q.strip()) == 0:
        return JSONResponse({
            "answer": "Пожалуйста, введите ваш вопрос.",
            "sources": ""
        })

    try:
        # Очищаем старые сессии
        clean_old_sessions()

        # Управление сессией
        if not session_id:
            session_id = str(uuid.uuid4())
            if response:
                response.set_cookie(key="session_id", value=session_id, max_age=SESSION_MAX_AGE)
            print(f"Создана новая сессия: {session_id}")
        else:
            print(f"Использована существующая сессия: {session_id}")
            if response:
                response.set_cookie(key="session_id", value=session_id, max_age=SESSION_MAX_AGE)

        # История диалога
        if session_id not in session_memories:
            session_memories[session_id] = []

        session_last_activity[session_id] = time.time()
        chat_history = session_memories[session_id]

        # Логируем текущую историю чата
        if chat_history:
            print(f"История диалога для сессии {session_id} (всего {len(chat_history)} обменов):")
            for i, (question, answer) in enumerate(chat_history[-3:]):  # Выводим последние 3 обмена для краткости
                print(f"  {i + 1}. Вопрос: {question[:50]}...")
                print(f"     Ответ: {answer[:50]}...")

        # Проверяем API ключи
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        if not openai_api_key:
            print("ОШИБКА: Ключ API OpenAI не найден в переменных окружения")
            return JSONResponse({
                "answer": "Ошибка: Не найден ключ API OpenAI для эмбеддингов. Пожалуйста, проверьте настройки .env файла.",
                "sources": ""
            }, status_code=500)

        if not anthropic_api_key:
            print("ОШИБКА: Ключ API Anthropic не найден в переменных окружения")
            return JSONResponse({
                "answer": "Ошибка: Не найден ключ API Anthropic для Claude. Пожалуйста, проверьте настройки .env файла.",
                "sources": ""
            }, status_code=500)

        # Проверка валидности ключа OpenAI для эмбеддингов
        try:
            print("Проверка API ключа OpenAI...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            # Маленький тест для проверки работоспособности API
            _ = embeddings.embed_query("тестовый запрос")
            print("API ключ OpenAI валиден")
        except Exception as e:
            error_msg = f"Ошибка API OpenAI: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return JSONResponse({
                "answer": f"Извините, возникла проблема с сервисом OpenAI для поиска. Пожалуйста, попробуйте позже.",
                "sources": ""
            }, status_code=500)

        # Загружаем индекс с дополнительными проверками
        try:
            print("Загрузка векторного хранилища...")
            vectorstore = load_vectorstore()
            print("Векторное хранилище успешно загружено")
        except Exception as e:
            error_msg = f"Ошибка загрузки индекса: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return JSONResponse({
                "answer": "Извините, произошла ошибка при доступе к базе знаний. Пожалуйста, попробуйте позже.",
                "sources": ""
            }, status_code=500)

        # Подготовка контекста из истории диалога
        dialog_context = ""
        if chat_history:
            dialog_context = "История диалога:\n"
            for i, (prev_q, prev_a) in enumerate(chat_history):
                dialog_context += f"Вопрос пользователя: {prev_q}\nТвой ответ: {prev_a}\n\n"

        # Обогащенный запрос с контекстом
        recent_dialogue = " ".join([qa[0] for qa in chat_history[-2:]]) if chat_history else ""
        enhanced_query = f"{recent_dialogue} {q}" if recent_dialogue else q

        # Получаем релевантные документы с обработкой исключений
        try:
            print(f"Выполняется поиск по запросу: '{enhanced_query[:50]}...'")

            # Используем MMR с оптимизированными параметрами для баланса
            # между релевантностью и разнообразием
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,  # Общее количество результатов
                    "fetch_k": 12,  # Уменьшено с 20 до 12 для большей точности
                    "lambda_mult": 0.7  # Увеличено с 0.5 до 0.7 для большего акцента на релевантности
                }
            )

            relevant_docs = retriever.get_relevant_documents(enhanced_query)
            print(f"Найдено {len(relevant_docs)} релевантных документов")

            # Вывод метаданных первого документа для диагностики
            if relevant_docs:
                doc_metadata = relevant_docs[0].metadata
                print(f"Пример метаданных документа: {doc_metadata}")
        except Exception as e:
            error_msg = f"Ошибка при поиске документов: {str(e)}"
            print(error_msg)
            traceback.print_exc()

            # Пробуем продолжить без документов
            relevant_docs = []
            print("Продолжаем работу без документов...")

        # Готовим контекст для LLM
        if len(relevant_docs) == 0:
            context = "Документов не найдено. Постарайся ответить, используя только историю диалога, если это возможно."
        else:
            context = ""
            for i, doc in enumerate(relevant_docs):
                context += f"Документ {i + 1}: {doc.page_content}\n\n"

        # Системный промпт, оптимизированный для Claude
        system_prompt = """
        Ты ассистент с доступом к базе знаний по финансовым и юридическим документам. 
        Используй информацию из базы знаний для ответа на вопросы.

        ОЧЕНЬ ВАЖНО: При ответе обязательно учитывай историю диалога и предыдущие вопросы пользователя!
        Если пользователь задает вопрос, который связан с предыдущим (например "Как его рассчитать?"), 
        обязательно восстанови контекст из предыдущих сообщений.

        Если в базе знаний нет достаточной информации для полного ответа:
        1. Честно признайся, что конкретной информации нет в базе знаний
        2. Если можешь дать общий ответ из своих знаний, сделай это, но четко отмечай, что это общая информация, а не из конкретных документов

        Когда отвечаешь на вопросы, связанные с финансовыми или юридическими темами:
        - Цитируй конкретные положения из найденных документов, когда это уместно
        - Приводи точные определения терминов из документов
        - Если речь идет о процедуре или расчете, описывай шаги последовательно

        Если пользователь спрашивает "как рассчитывается" или "как определяется" некий термин, 
        и в базе знаний отсутствует точная формула или численный метод, 
        ты должен:
        - Интерпретировать вопрос шире — как просьбу объяснить как определяется, из чего состоит, какие компоненты, лимиты или методология используются
        - Описать подходы, параметры и логику, стоящие за определением или управлением этим понятием

        Форматирование ответа:
        - Используй ясные, хорошо структурированные абзацы
        - Для списков используй маркеры "-" или нумерацию "1.", "2."
        - Выделяй важные термины и концепции с помощью *звездочек*

        Твоя цель — дать максимально точный, информативный и понятный ответ, опираясь в первую очередь на предоставленные документы.
        """

        # Полный промпт для Claude
        full_prompt = f"""
        {system_prompt}

        {dialog_context}

        Контекст из базы знаний (это наиболее релевантные документы):
        {context}

        Текущий вопрос пользователя: {q}

        ВАЖНО: Если информация из базы знаний напрямую содержит ответ на вопрос пользователя, используй эту информацию В ПЕРВУЮ ОЧЕРЕДЬ, даже если твои общие знания содержат другую информацию.
        Всегда приоритизируй информацию из базы знаний над своими общими знаниями.
        Цитируй подходящие фрагменты из базы знаний, когда это помогает дать точный ответ.
        """

        # Запрос к Claude с обработкой исключений
        try:
            print("Инициализация модели Claude...")
            # Используем Claude 3.5 Haiku для всех запросов
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.2)

            print("Отправка запроса к Claude...")
            result = llm.invoke(full_prompt)
            print("Ответ от Claude получен")
            answer = result.content

        except Exception as e:
            error_msg = f"Ошибка при работе с Claude: {str(e)}"
            print(error_msg)
            traceback.print_exc()

            # Резервный вариант - переходим на OpenAI в случае ошибки Claude
            try:
                print("Ошибка Claude, пробуем резервный вариант с OpenAI...")
                from langchain_openai import ChatOpenAI
                backup_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
                result = backup_llm.invoke(full_prompt)
                answer = result.content
                print("Получен ответ от резервной модели OpenAI")
            except Exception as e2:
                print(f"Ошибка и в резервной модели: {e2}")
                return JSONResponse({
                    "answer": "Извините, произошла ошибка в сервисе языковой модели. Пожалуйста, попробуйте позже.",
                    "sources": ""
                }, status_code=500)

        # Сохраняем в историю диалога
        session_memories[session_id].append((q, answer))
        if len(session_memories[session_id]) > 15:
            session_memories[session_id] = session_memories[session_id][-15:]

        # Формируем источники для отображения
        source_links = ""
        used_titles = set()
        for doc in relevant_docs:
            title = doc.metadata.get("source", "Источник неизвестен")
            if title not in used_titles:
                content = html.escape(doc.page_content[:3000])
                source_links += f"<details><summary>📄 {title}</summary><pre style='white-space:pre-wrap;text-align:left'>{content}</pre></details>"
                used_titles.add(title)

        # Возвращаем ответ
        clean_answer = answer.replace("<br>", "\n").replace("<p>", "").replace("</p>", "\n")
        return JSONResponse({"answer": clean_answer, "sources": source_links})

    except Exception as e:
        error_message = f"Ошибка при обработке запроса: {str(e)}"
        print(error_message)
        print(f"Тип ошибки: {type(e).__name__}")
        traceback.print_exc()  # Выводит полный стек ошибки

        # Запись ошибки в лог
        log_dir = "/data"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "error.log")

        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"=== Ошибка запроса от {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log.write(f"Вопрос: {q}\n")
            log.write(f"Ошибка: {error_message}\n")
            log.write(f"Трассировка:\n{traceback.format_exc()}\n\n")

        return JSONResponse({
            "answer": f"Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже или обратитесь к администратору.",
            "sources": ""
        }, status_code=500)