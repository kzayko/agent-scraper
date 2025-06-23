import os
import logging
from flask import Flask, request, jsonify, render_template, render_template_string
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

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

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
