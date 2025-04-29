"""
Модуль для интеграции RAG чат-бота с Telegram.
Использует общую логику обработки запросов из main.py,
но предоставляет интерфейс для взаимодействия через Telegram.
"""

import os
import logging
import asyncio
import threading
from datetime import datetime
import traceback
import signal

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# Импортируем функции из основного модуля
from main import load_vectorstore, session_memories, session_last_activity, SESSION_MAX_AGE, clean_old_sessions, \
    telegram_sessions

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Версия модуля
__version__ = "1.0.2"

# Глобальная переменная для хранения загруженного индекса
vectorstore = None

# Глобальная переменная для хранения экземпляра приложения
application = None

# Максимальное количество токенов в ответе (для разбиения длинных ответов)
MAX_MESSAGE_LENGTH = 4096


def is_greeting(text):
    """Проверяет, является ли текст приветствием."""
    greetings = [
        "привет", "здравствуй", "здравствуйте", "добрый день", "доброе утро",
        "добрый вечер", "hi", "hello", "hey", "приветствую", "здарова",
        "салют", "хай", "хеллоу"
    ]
    text_lower = text.lower()
    # Проверяем, состоит ли весь текст только из приветствия
    # или текст начинается с приветствия и содержит не более 3 слов
    return (text_lower in greetings or
            any(text_lower.startswith(g) for g in greetings) and len(text_lower.split()) <= 3)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    user_id = update.effective_user.id

    # Создаем новую сессию для пользователя
    session_id = str(user_id)
    if session_id not in session_memories:
        session_memories[session_id] = []
    session_last_activity[session_id] = datetime.now().timestamp()
    telegram_sessions[user_id] = session_id

    await update.message.reply_text(
        "👋 Добро пожаловать! Я чат-бот с доступом к базе знаний по финансовым и юридическим документам.\n\n"
        "Задайте мне вопрос, и я найду релевантную информацию.\n\n"
        "Команды:\n"
        "/start - Начать новую беседу\n"
        "/help - Справка\n"
        "/clear - Очистить историю диалога"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    await update.message.reply_text(
        "🔍 Я - RAG чат-бот, использующий базу знаний для ответов на ваши вопросы.\n\n"
        "Как со мной работать:\n"
        "- Просто задайте вопрос, и я найду ответ в базе знаний\n"
        "- Я помню историю диалога и могу продолжать обсуждение\n"
        "- Я работаю с финансовыми и юридическими документами\n\n"
        "Команды:\n"
        "/start - Начать новую беседу\n"
        "/clear - Очистить историю диалога\n"
        "/help - Показать эту справку"
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /clear для очистки истории диалога."""
    user_id = update.effective_user.id

    if user_id in telegram_sessions:
        session_id = telegram_sessions[user_id]
        if session_id in session_memories:
            session_memories[session_id] = []
            session_last_activity[session_id] = datetime.now().timestamp()
            await update.message.reply_text("🧹 История диалога очищена!")
        else:
            await update.message.reply_text("История диалога уже пуста.")
    else:
        # Создаем новую сессию
        session_id = str(user_id)
        session_memories[session_id] = []
        session_last_activity[session_id] = datetime.now().timestamp()
        telegram_sessions[user_id] = session_id
        await update.message.reply_text("🧹 История диалога очищена!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Основной обработчик текстовых сообщений."""
    global vectorstore

    # Получаем идентификатор пользователя
    user_id = update.effective_user.id
    user_input = update.message.text

    # Проверка на чистый символ "/", который иногда отправляется случайно
    if user_input.strip() == "/":
        return

    # Проверяем, если это команда (начинается с /), но не обрабатываемая напрямую
    if user_input.startswith('/') and len(user_input) > 1:
        await update.message.reply_text(
            "❓ Неизвестная команда. Доступные команды:\n"
            "/start - Начать новую беседу\n"
            "/help - Показать справку\n"
            "/clear - Очистить историю диалога"
        )
        return

    # Получаем сессию для пользователя
    if user_id not in telegram_sessions:
        session_id = str(user_id)
        telegram_sessions[user_id] = session_id
    else:
        session_id = telegram_sessions[user_id]

    # Создаем сессию, если она не существует
    if session_id not in session_memories:
        session_memories[session_id] = []

    # Обновляем время последней активности
    session_last_activity[session_id] = datetime.now().timestamp()

    # Очищаем старые сессии для экономии памяти
    clean_old_sessions()

    # Отправляем уведомление о начале обработки запроса
    processing_message = await update.message.reply_text("🔍 Ищу ответ на ваш вопрос...")

    try:
        # Проверка на приветствие перед запросом к базе знаний
        if is_greeting(user_input):
            await processing_message.delete()  # Удаляем сообщение о поиске
            greeting_response = f"Здравствуйте! Чем я могу вам помочь? Задайте вопрос по финансовым или юридическим темам."

            # Сохраняем приветствие в историю диалога
            session_memories[session_id].append((user_input, greeting_response))
            if len(session_memories[session_id]) > 15:
                session_memories[session_id] = session_memories[session_id][-15:]

            await update.message.reply_text(greeting_response)
            return  # Завершаем обработку сообщения

        # Ленивая загрузка индекса при первом запросе
        if vectorstore is None:
            try:
                await processing_message.edit_text("🔄 Загружаю базу знаний... Это может занять некоторое время.")
                # Загружаем индекс асинхронно
                vectorstore = await asyncio.to_thread(load_vectorstore)
            except Exception as e:
                logger.error(f"Ошибка при загрузке индекса: {e}")
                await processing_message.edit_text(
                    "❌ Ошибка при загрузке базы знаний. Пожалуйста, попробуйте позже."
                )
                return

        # Формируем контекст из истории диалога
        chat_history = session_memories[session_id]

        # Делаем запрос к базе знаний
        recent_dialogue = " ".join([qa[0] for qa in chat_history[-2:]]) if chat_history else ""
        enhanced_query = f"{recent_dialogue} {user_input}" if recent_dialogue else user_input

        # Выполняем поиск асинхронно, чтобы не блокировать основной поток
        try:
            # Используем новый метод invoke вместо устаревшего get_relevant_documents
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 12,
                    "lambda_mult": 0.7
                }
            )

            # Используем новый формат вызова через invoke если доступен
            try:
                # Пытаемся использовать новый метод invoke
                relevant_docs = await asyncio.to_thread(
                    lambda q: retriever.invoke(q),
                    enhanced_query
                )
            except (AttributeError, TypeError):
                # Если не удалось, используем устаревший метод (но с предупреждением)
                logger.warning("Использование устаревшего метода get_relevant_documents. Рекомендуется обновить код.")
                relevant_docs = await asyncio.to_thread(
                    retriever.get_relevant_documents,
                    enhanced_query
                )

        except Exception as e:
            logger.error(f"Ошибка при поиске документов: {e}")
            traceback.print_exc()
            relevant_docs = []

        # Подготовка контекста для модели
        if len(relevant_docs) == 0:
            context = "Документов не найдено. Постарайся ответить, используя только историю диалога, если это возможно."
        else:
            context = ""
            for i, doc in enumerate(relevant_docs):
                doc_title = doc.metadata.get("source", f"Документ {i + 1}")
                if len(doc_title) > 100:
                    doc_title = doc_title[:100] + "..."
                context += f"{doc_title}: {doc.page_content}\n\n"

        await processing_message.edit_text("🧠 Генерирую ответ...")

        # Подготовка системного промпта и запроса к модели LLM
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

        При цитировании источников информации:
        - Используй реальные названия документов вместо "Документ 1", "Документ 2" и т.д.
        - Можешь сокращать длинные названия, сохраняя ключевую идентифицирующую информацию
        - При ссылке на конкретный источник используй формат: "Согласно [название документа], ..."

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

        # Формируем контекст из истории диалога
        dialog_context = ""
        if chat_history:
            dialog_context = "История диалога:\n"
            for i, (prev_q, prev_a) in enumerate(chat_history):
                dialog_context += f"Вопрос пользователя: {prev_q}\nТвой ответ: {prev_a}\n\n"

        # Полный промпт для модели с указанием формата ответа
        full_prompt = f"""
        {system_prompt}

        {dialog_context}

        Контекст из базы знаний (это наиболее релевантные документы):
        {context}

        Текущий вопрос пользователя: {user_input}

        ВАЖНО: 
        1. Начни свой ответ со слов "ОТВЕТ:" и сразу переходи к сути.
        2. В начале ответа ОБЯЗАТЕЛЬНО напиши фразу "В соответствии с [название документа]," и далее приведи конкретную информацию из документа.
        3. Не предоставляй никаких вводных объяснений о том, как ты будешь отвечать или какие источники использовать.
        4. Обязательно указывай названия документов при цитировании, используя точный формат источников из контекста.
        5. Если в базе знаний нет информации по вопросу, начни ответ с "ОТВЕТ: В доступных документах не найдена информация о..."
        """

        # Запрос к модели LLM асинхронно
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.2)

            # Асинхронно получаем ответ от модели
            result = await asyncio.to_thread(llm.invoke, full_prompt)
            answer = result.content

        except Exception as e:
            logger.error(f"Ошибка при работе с Claude: {e}")

            # Резервный вариант - переходим на OpenAI в случае ошибки Claude
            try:
                logger.info("Ошибка Claude, пробуем резервный вариант с OpenAI...")
                from langchain_openai import ChatOpenAI
                backup_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
                result = await asyncio.to_thread(backup_llm.invoke, full_prompt)
                answer = result.content
            except Exception as e2:
                logger.error(f"Ошибка и в резервной модели: {e2}")
                await processing_message.edit_text(
                    "❌ Извините, произошла ошибка в сервисе языковой модели. Пожалуйста, попробуйте позже."
                )
                return

        # Обработка полученного ответа для удаления маркера "ОТВЕТ:"
        if "ОТВЕТ:" in answer:
            answer = answer.split("ОТВЕТ:", 1)[1].strip()

        # Сохраняем в историю диалога
        session_memories[session_id].append((user_input, answer))
        if len(session_memories[session_id]) > 15:
            session_memories[session_id] = session_memories[session_id][-15:]

        # Удаляем сообщение о загрузке
        await processing_message.delete()

        # Разбиваем длинный ответ на части, если необходимо
        if len(answer) <= MAX_MESSAGE_LENGTH:
            await update.message.reply_text(answer)
        else:
            # Разбиваем на части по MAX_MESSAGE_LENGTH символов
            chunks = [answer[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(answer), MAX_MESSAGE_LENGTH)]
            for i, chunk in enumerate(chunks):
                await update.message.reply_text(f"Часть {i + 1}/{len(chunks)}:\n\n{chunk}")

        # Добавляем кнопку для показа источников, если они есть
        if relevant_docs:
            # Сохраняем релевантные документы в контексте пользователя для показа источников
            user_id_str = str(user_id)

            # Создаем хранилище для источников, если оно не существует
            if not hasattr(context, 'bot_data'):
                context.bot_data = {}
            if 'user_sources' not in context.bot_data:
                context.bot_data['user_sources'] = {}

            # Сохраняем только имена документов и короткие фрагменты содержимого
            context.bot_data['user_sources'][user_id_str] = []
            for doc in relevant_docs:
                title = doc.metadata.get("source", "Источник неизвестен")
                content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                context.bot_data['user_sources'][user_id_str].append({
                    "title": title,
                    "content": content_preview
                })

            keyboard = [
                [InlineKeyboardButton("📚 Показать источники",
                                      callback_data=f"sources_{user_id}_{datetime.now().timestamp()}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("Нажмите для просмотра источников ответа:", reply_markup=reply_markup)

    except Exception as e:
        error_message = f"Произошла ошибка при обработке запроса: {str(e)}"
        logger.error(error_message)
        logger.error(f"Трассировка: {traceback.format_exc()}")

        # Сохраняем ошибку в лог
        log_dir = "/data"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "telegram_error.log")

        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"=== Ошибка запроса от {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log.write(f"Пользователь: {user_id}\n")
            log.write(f"Вопрос: {user_input}\n")
            log.write(f"Ошибка: {error_message}\n")
            log.write(f"Трассировка:\n{traceback.format_exc()}\n\n")

        await processing_message.edit_text(
            "❌ Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
        )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик нажатий на кнопки."""
    query = update.callback_query
    await query.answer()

    # Получаем данные колбэка
    callback_data = query.data

    if callback_data.startswith("sources_"):
        # Получаем ID пользователя из данных колбэка
        parts = callback_data.split("_")
        user_id = int(parts[1])

        # Проверяем, что пользователь, нажавший кнопку, тот же, кто задал вопрос
        if user_id == query.from_user.id:
            user_id_str = str(user_id)

            # Проверяем, сохранены ли источники для этого пользователя
            try:
                if hasattr(context, 'bot_data') and 'user_sources' in context.bot_data and user_id_str in \
                        context.bot_data['user_sources']:
                    sources = context.bot_data['user_sources'][user_id_str]
                    if sources:
                        # Формируем сообщение с источниками
                        sources_text = "📚 Использованные источники:\n\n"
                        for i, source in enumerate(sources, 1):
                            sources_text += f"{i}. {source['title']}\n"
                            sources_text += f"Фрагмент: {source['content'][:200]}...\n\n"

                        # Отправляем без Markdown для избежания ошибок форматирования
                        await query.message.reply_text(sources_text)
                    else:
                        await query.message.reply_text("Источники для этого ответа не сохранены.")
                else:
                    await query.message.reply_text("Источники для этого ответа недоступны.")
            except Exception as e:
                logger.error(f"Ошибка при показе источников: {e}")
                await query.message.reply_text("Произошла ошибка при попытке показать источники.")
        else:
            await query.message.reply_text("Эта кнопка предназначена только для автора вопроса.")


def create_telegram_bot():
    """Создает и настраивает Telegram бота."""
    global application

    # Получаем токен бота из переменных окружения
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN не найден в переменных окружения")
        return None

    try:
        # Создаем приложение
        builder = Application.builder().token(bot_token)
        # Установка более корректной работы с сигналами завершения
        builder.post_stop(shutdown_handler)
        application = builder.build()

        # Добавляем обработчики команд
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("clear", clear_command))

        # Обработчик текстовых сообщений
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        # Обработчик нажатий на кнопки
        application.add_handler(CallbackQueryHandler(button_callback))

        logger.info("Telegram бот успешно настроен")
        return application
    except Exception as e:
        logger.error(f"Ошибка при создании Telegram бота: {e}")
        traceback.print_exc()
        return None


async def shutdown_handler():
    """Корректное завершение работы бота при остановке приложения."""
    logger.info("Завершение работы Telegram бота...")


def telegram_bot_thread_function():
    """Функция для запуска бота в отдельном потоке."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Создаем новый цикл событий для этого потока
        bot_app = create_telegram_bot()
        if not bot_app:
            logger.error("Не удалось создать приложение Telegram бота.")
            return

        # Запускаем бота
        loop.run_until_complete(bot_app.initialize())
        loop.run_until_complete(bot_app.start())
        loop.run_until_complete(bot_app.updater.start_polling())

        logger.info("Telegram бот запущен и ожидает сообщений")

        # Вместо добавления обработчиков сигналов, которые работают только в основном потоке,
        # просто запускаем бесконечный цикл
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            # Это исключение будет перехвачено только в основном потоке,
            # но мы включаем его для полноты
            pass
    except Exception as e:
        logger.error(f"Ошибка в потоке Telegram бота: {e}")
        traceback.print_exc()
    finally:
        # Обеспечиваем правильное закрытие цикла событий при выходе
        try:
            if 'bot_app' in locals():
                loop.run_until_complete(bot_app.stop())
                loop.run_until_complete(bot_app.shutdown())
            loop.close()
        except Exception as e:
            logger.error(f"Ошибка при завершении бота: {e}")
        logger.info("Поток Telegram бота завершен")


# Эта функция будет вызываться из main.py
def start_telegram_bot():
    """Запускает Telegram бота в отдельном потоке."""
    if not os.getenv("TELEGRAM_BOT_TOKEN"):
        logger.warning("TELEGRAM_BOT_TOKEN не найден. Telegram бот не будет запущен.")
        return

    # Запускаем бота в отдельном потоке с выделенным циклом событий
    bot_thread = threading.Thread(target=telegram_bot_thread_function, daemon=True)
    bot_thread.start()
    logger.info("Telegram бот запущен в отдельном потоке")