
import os
import logging
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import threading
import time
import config
from agent import get_agent
from utils.logger import get_logger, WebLogHandler

# Настройка логирования
app_logger = get_logger("webapp")
agent_logger = get_logger("agent")

# Создание Flask приложения
app = Flask(__name__)
CORS(app)

# Глобальные переменные для отслеживания процесса
processing_completed = False
current_logs = []
log_handler = None

# HTML шаблон для веб-интерфейса
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Агент-суммаризатор информации с GigaChat</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }

        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-style: italic;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #34495e;
        }

        textarea, input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
            font-family: inherit;
            box-sizing: border-box;
        }

        textarea {
            min-height: 120px;
            resize: vertical;
        }

        button {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
            width: 100%;
        }

        button:hover:not(:disabled) {
            background: linear-gradient(135deg, #2980b9, #1f5582);
            transform: translateY(-1px);
        }

        button:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
        }

        .status {
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
            font-weight: bold;
            text-align: center;
        }

        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f1aeb5; }
        .status.processing { background: #d1ecf1; color: #0c5460; border: 1px solid #b8daff; }

        .logs-container {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-size: 14px;
        }

        .result-container {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
            border-left: 4px solid #28a745;
        }

        .result-container h3 {
            color: #28a745;
            margin-top: 0;
        }

        .result-text {
            background: white;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            white-space: pre-wrap;
            line-height: 1.6;
        }

        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .feature-info {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            border-left: 4px solid #17a2b8;
        }

        .feature-info h4 {
            margin-top: 0;
            color: #17a2b8;
        }

        .statistics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .stat-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            border: 1px solid #dee2e6;
        }

        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }

        .stat-label {
            color: #6c757d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Агент-суммаризатор информации</h1>
        <p class="subtitle">Работает на основе GigaChat с векторной БД Qdrant</p>

        <div class="feature-info">
            <h4>🔧 Новые возможности:</h4>
            <ul>
                <li><strong>GigaChat Embeddings:</strong> Использует embeddings модель GigaChat для векторизации текста</li>
                <li><strong>Username/Password авторизация:</strong> Подключение к GigaChat через логин и пароль</li>
                <li><strong>Улучшенная точность:</strong> Более качественный поиск релевантной информации</li>
                <li><strong>Русскоязычная оптимизация:</strong> Специально настроен для работы с русским языком</li>
            </ul>
        </div>

        <form id="queryForm">
            <div class="form-group">
                <label for="user_query">Введите ваш запрос:</label>
                <textarea 
                    id="user_query" 
                    name="user_query" 
                    placeholder="Например: Расскажи о современных методах машинного обучения и их применении в обработке естественного языка"
                    required
                ></textarea>
            </div>

            <div class="form-group">
                <label for="sources_file">Загрузите Excel файл с источниками (колонка 'url' или 'URL'):</label>
                <input 
                    type="file" 
                    id="sources_file" 
                    name="sources_file" 
                    accept=".xlsx,.xls"
                    required
                >
            </div>

            <button type="submit" id="submitButton">
                <span id="buttonText">🚀 Запустить анализ</span>
            </button>
        </form>

        <div id="status" class="status" style="display: none;"></div>

        <div id="logsContainer" style="display: none;">
            <h3>📋 Логи выполнения:</h3>
            <div id="logs" class="logs-container"></div>
        </div>

        <div id="resultContainer" style="display: none;">
            <div class="result-container">
                <h3>📊 Результаты анализа</h3>
                <div id="statistics" class="statistics"></div>
                <div id="result" class="result-text"></div>
            </div>
        </div>
    </div>

    <script>
        let processingInterval;

        document.getElementById('queryForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const formData = new FormData();
            formData.append('user_query', document.getElementById('user_query').value);

            const fileInput = document.getElementById('sources_file');
            if (fileInput.files.length > 0) {
                formData.append('sources_file', fileInput.files[0]);
            }

            // Обновляем интерфейс
            updateStatus('processing', '⏳ Обработка запроса...');
            document.getElementById('submitButton').disabled = true;
            document.getElementById('buttonText').innerHTML = '<span class="spinner"></span>Обработка...';
            document.getElementById('logsContainer').style.display = 'block';
            document.getElementById('resultContainer').style.display = 'none';

            // Запускаем отслеживание логов
            startLogPolling();

            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    updateStatus('success', '✅ Анализ завершен успешно!');
                    displayResult(result.data);
                } else {
                    updateStatus('error', `❌ Ошибка: ${result.error}`);
                }
            } catch (error) {
                updateStatus('error', `❌ Ошибка сети: ${error.message}`);
            } finally {
                document.getElementById('submitButton').disabled = false;
                document.getElementById('buttonText').innerHTML = '🚀 Запустить анализ';
                stopLogPolling();
            }
        });

        function updateStatus(type, message) {
            const statusElement = document.getElementById('status');
            statusElement.className = `status ${type}`;
            statusElement.textContent = message;
            statusElement.style.display = 'block';
        }

        function startLogPolling() {
            processingInterval = setInterval(async () => {
                try {
                    const response = await fetch('/api/logs');
                    const logs = await response.json();
                    document.getElementById('logs').textContent = logs.join('\n');

                    // Автоскролл вниз
                    const logsElement = document.getElementById('logs');
                    logsElement.scrollTop = logsElement.scrollHeight;
                } catch (error) {
                    console.error('Ошибка получения логов:', error);
                }
            }, 1000);
        }

        function stopLogPolling() {
            if (processingInterval) {
                clearInterval(processingInterval);
            }
        }

        function displayResult(data) {
            // Отображение статистики
            const statisticsHTML = `
                <div class="stat-item">
                    <div class="stat-number">${data.processed_sources}</div>
                    <div class="stat-label">Обработано источников</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">${data.total_documents}</div>
                    <div class="stat-label">Всего блоков</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">${data.questions.length}</div>
                    <div class="stat-label">Сгенерировано вопросов</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">${data.question_answers.length}</div>
                    <div class="stat-label">Ответов получено</div>
                </div>
            `;

            document.getElementById('statistics').innerHTML = statisticsHTML;
            document.getElementById('result').textContent = data.final_report;
            document.getElementById('resultContainer').style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Главная страница"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/process', methods=['POST'])
def process_request():
    """API для обработки запроса"""
    global processing_completed, current_logs, log_handler

    processing_completed = False
    current_logs = []

    try:
        user_query = request.form.get('user_query')
        if not user_query:
            return jsonify({'success': False, 'error': 'Не указан запрос пользователя'}), 400

        # Обработка загружаемого файла
        sources_file = request.files.get('sources_file')
        if sources_file and sources_file.filename.endswith(('.xlsx', '.xls')):
            sources_file.save(config.SOURCES_EXCEL_PATH)
            app_logger.info(f"Загружен файл источников: {sources_file.filename}")
        elif not os.path.exists(config.SOURCES_EXCEL_PATH):
            return jsonify({
                'success': False, 
                'error': f'Файл источников не найден: {config.SOURCES_EXCEL_PATH}'
            }), 400

        # Настройка обработчика логов для веб-интерфейса
        log_handler = WebLogHandler(current_logs)
        agent_logger.addHandler(log_handler)

        app_logger.info(f"Начало обработки запроса: {user_query}")

        # Получение агента и обработка запроса
        agent = get_agent()
        result = agent.process_query(user_query)

        processing_completed = True

        if result['status'] == 'success':
            app_logger.info("Запрос обработан успешно")
            return jsonify({'success': True, 'data': result})
        else:
            app_logger.error(f"Ошибка при обработке запроса: {result.get('error', 'Неизвестная ошибка')}")
            return jsonify({'success': False, 'error': result.get('error', 'Неизвестная ошибка')}), 500

    except Exception as e:
        app_logger.error(f"Ошибка обработки API запроса: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        # Удаляем обработчик логов
        if log_handler:
            agent_logger.removeHandler(log_handler)
            log_handler = None

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """API для получения логов"""
    return jsonify(current_logs)

@app.route('/api/status', methods=['GET'])
def get_status():
    """API для получения статуса обработки"""
    return jsonify({'completed': processing_completed})

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка состояния сервиса"""
    try:
        # Проверяем подключение к Qdrant
        from utils.vector_db import VectorDatabase
        vector_db = VectorDatabase()
        qdrant_info = vector_db.get_collection_info()

        return jsonify({
            'status': 'healthy',
            'gigachat': 'configured',
            'qdrant': 'connected',
            'qdrant_info': qdrant_info
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

def run_web_app():
    """Запуск веб-приложения"""
    app_logger.info(f"Запуск веб-приложения на {config.WEB_HOST}:{config.WEB_PORT}")
    app.run(
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        debug=config.WEB_DEBUG,
        threaded=True
    )

if __name__ == '__main__':
    run_web_app()
