// Шорткод для бота с интеграцией Login/Signup Popup
function rag_bot_with_popup_shortcode() {
    ob_start();
    // Проверяем, активирован ли плагин Login/Signup Popup
    if (!function_exists('lsphe_popup_login_enqueue')) {
        echo '<div style="color:red;">Плагин Login/Signup Popup не активирован!</div>';
        return ob_get_clean();
    }

    // Определяем, авторизован ли пользователь
    $is_logged_in = is_user_logged_in();
    ?>
    <div class="rag-bot-container">
        <div class="rag-bot-header">
            <h2>Онлайн помощник</h2>
            <?php if ($is_logged_in):
                $current_user = wp_get_current_user();
            ?>
                <div class="rag-logged-in-info">
                    <span>Вы вошли как: <?php echo esc_html($current_user->display_name); ?></span>
                </div>
            <?php endif; ?>
        </div>

        <div class="rag-chat-area">
            <div class="rag-messages" id="ragMessages">
                <div class="rag-welcome-message">
                    <h3>Добро пожаловать в онлайн-помощника!</h3>
                    <p>Задайте вопрос, и я постараюсь помочь.</p>
                </div>
            </div>

            <form id="ragQuestionForm" class="rag-question-form">
                <input type="text" id="ragQuestionInput" placeholder="Введите ваш вопрос..." required>
                <button type="submit" id="ragSubmitBtn">Отправить</button>
            </form>
        </div>
    </div>

    <style>
    .rag-bot-container {
        max-width: 700px;
        margin: 0 auto;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .rag-bot-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 20px;
        background-color: #f8f9fa;
        border-bottom: 1px solid #e0e0e0;
    }
    .rag-bot-header h2 {
        margin: 0;
        color: #333;
    }
    .rag-logged-in-info {
        font-size: 14px;
        color: #666;
    }
    .rag-chat-area {
        display: flex;
        flex-direction: column;
    }
    .rag-messages {
        height: 400px;
        overflow-y: auto;
        padding: 20px;
        background-color: #fff;
    }
    .rag-welcome-message {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        text-align: center;
    }
    .rag-welcome-message h3 {
        margin-top: 0;
        color: #0073aa;
    }
    .rag-user-message, .rag-bot-message {
        padding: 12px 15px;
        border-radius: 18px;
        margin-bottom: 15px;
        max-width: 80%;
    }
    .rag-user-message {
        background-color: #0073aa;
        color: white;
        margin-left: auto;
    }
    .rag-bot-message {
        background-color: #f1f1f1;
        margin-right: auto;
    }
    .rag-question-form {
        display: flex;
        padding: 15px;
        border-top: 1px solid #e0e0e0;
    }
    #ragQuestionInput {
        flex: 1;
        padding: 12px 15px;
        border: 1px solid #ddd;
        border-radius: 30px;
        font-size: 15px;
        margin-right: 10px;
    }
    #ragSubmitBtn {
        background-color: #0073aa;
        color: white;
        border: none;
        padding: 0 20px;
        border-radius: 30px;
        cursor: pointer;
        font-size: 15px;
        font-weight: 500;
    }
    #ragSubmitBtn:hover {
        background-color: #005d87;
    }
    .rag-loading {
        display: flex;
        align-items: center;
        margin-right: auto;
        background-color: #f1f1f1;
        padding: 10px 15px;
        border-radius: 18px;
        margin-bottom: 15px;
    }
    .rag-loading-dots span {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #666;
        border-radius: 50%;
        margin-right: 5px;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    .rag-loading-dots span:nth-child(1) { animation-delay: -0.32s; }
    .rag-loading-dots span:nth-child(2) { animation-delay: -0.16s; }

    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1.0); }
    }
    </style>

    <script type="text/javascript">
    jQuery(document).ready(function($) {
        // Получаем AJAX URL WordPress
        var ajaxurl = '<?php echo admin_url('admin-ajax.php'); ?>';

        // Обработчик отправки формы
        $('#ragQuestionForm').on('submit', function(e) {
            e.preventDefault();

            // Получаем введенный вопрос
            var question = $('#ragQuestionInput').val().trim();
            if (!question) return;

            <?php if (!$is_logged_in): ?>
            // Для неавторизованных пользователей показываем всплывающее окно входа
            // Используем функции плагина Login/Signup Popup
            lsphe_open_login_popup();
            <?php else: ?>
            // Для авторизованных пользователей обрабатываем вопрос
            sendQuestionToBot(question);
            <?php endif; ?>
        });

        // Функция для отправки вопроса боту
        function sendQuestionToBot(question) {
            // Добавляем сообщение пользователя в чат
            $('#ragMessages').append('<div class="rag-user-message">' + question + '</div>');

            // Очищаем поле ввода
            $('#ragQuestionInput').val('');

            // Добавляем индикатор загрузки
            var loadingDiv = $('<div class="rag-loading">Получаем ответ <div class="rag-loading-dots"><span></span><span></span><span></span></div></div>');
            $('#ragMessages').append(loadingDiv);

            // Прокручиваем к последнему сообщению
            $('#ragMessages').scrollTop($('#ragMessages')[0].scrollHeight);

            // Отправляем AJAX запрос
            $.ajax({
                url: ajaxurl,
                type: 'POST',
                data: {
                    action: 'rag_bot_question',
                    question: question,
                    security: '<?php echo wp_create_nonce("rag-bot-security"); ?>'
                },
                success: function(response) {
                    // Удаляем индикатор загрузки
                    loadingDiv.remove();

                    if (response.success) {
                        // Добавляем ответ бота
                        $('#ragMessages').append('<div class="rag-bot-message">' + response.data + '</div>');
                    } else {
                        // Если произошла ошибка
                        $('#ragMessages').append('<div class="rag-bot-message">Произошла ошибка: ' + response.data + '</div>');
                    }

                    // Прокручиваем к последнему сообщению
                    $('#ragMessages').scrollTop($('#ragMessages')[0].scrollHeight);
                },
                error: function() {
                    // Удаляем индикатор загрузки
                    loadingDiv.remove();

                    // Добавляем сообщение об ошибке
                    $('#ragMessages').append('<div class="rag-bot-message">Произошла ошибка при соединении с сервером. Пожалуйста, попробуйте позже.</div>');

                    // Прокручиваем к последнему сообщению
                    $('#ragMessages').scrollTop($('#ragMessages')[0].scrollHeight);
                }
            });
        }

        // Если пользователь вошел после открытия страницы, перезагрузим страницу
        $(document).on('lsphe_user_logged_in', function() {
            location.reload();
        });
    });
    </script>
    <?php

    return ob_get_clean();
}
add_shortcode('rag_bot_popup', 'rag_bot_with_popup_shortcode');

// Обработчик AJAX для бота
function handle_rag_bot_question_ajax() {
    // Проверяем nonce
    check_ajax_referer('rag-bot-security', 'security');

    // Проверяем авторизацию
    if (!is_user_logged_in()) {
        wp_send_json_error('Необходима авторизация');
        return;
    }

    // Получаем вопрос
    $question = sanitize_text_field($_POST['question']);

    // Здесь ваш код для обработки вопроса и получения ответа от RAG-бота
    // Пример ответа (замените на реальный код для вашего бота):
    $answer = get_bot_response($question);

    wp_send_json_success($answer);
}
add_action('wp_ajax_rag_bot_question', 'handle_rag_bot_question_ajax');
add_action('wp_ajax_nopriv_rag_bot_question', function() {
    wp_send_json_error('Требуется авторизация');
});

// Функция для получения ответа от бота (заглушка)
function get_bot_response($question) {
    // Замените этот код на интеграцию с вашим RAG-ботом
    // Пример:
    /*
    $response = wp_remote_post('https://ваш-бот-api-url.com/query', array(
        'body' => json_encode(array(
            'question' => $question,
            'user_id' => get_current_user_id()
        )),
        'headers' => array(
            'Content-Type' => 'application/json',
            'Authorization' => 'Bearer ваш-api-ключ'
        )
    ));

    if (!is_wp_error($response)) {
        $body = wp_remote_retrieve_body($response);
        $data = json_decode($body, true);
        return isset($data['answer']) ? $data['answer'] : 'Извините, не удалось получить ответ.';
    }
    */

    // Заглушка для тестирования:
    return "Это ответ от бота на ваш вопрос: \"" . $question . "\". Здесь будет ответ от вашего RAG-бота.";
}