import os
import logging
from flask import Flask, request, jsonify, render_template, render_template_string, send_from_directory
from flask_cors import CORS
import threading
import time
import config
from agent import get_agent
from utils.logger import get_logger, WebLogHandler
import uuid
import json
from datetime import datetime
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

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

class WebFullLogHandler(logging.Handler):
    """Обработчик логов для веб-интерфейса (полный формат в отдельный файл)"""

    def __init__(self, log_file_path: str):
        super().__init__()
        self.log_file_path = log_file_path
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        # Создаём файл и записываем заголовок
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Логи веб-запроса {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    def emit(self, record):
        """Записывает полную запись лога в файл"""
        try:
            log_entry = self.format(record)
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception:
            self.handleError(record)

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

    temp_excel_path = None
    try:
        user_query = request.form.get('user_query')
        if not user_query:
            return jsonify({'success': False, 'error': 'Не указан запрос пользователя'}), 400

        # Обработка загружаемого файла
        sources_file = request.files.get('sources_file')
        if sources_file and sources_file.filename.endswith(('.xlsx', '.xls')):
            os.makedirs('results', exist_ok=True)
            temp_excel_path = f"results/sources_{uuid.uuid4().hex}.xlsx"
            sources_file.save(temp_excel_path)
            app_logger.info(f"Загружен файл источников: {sources_file.filename} -> {temp_excel_path}")
        elif not os.path.exists(config.SOURCES_EXCEL_PATH):
            return jsonify({
                'success': False, 
                'error': f'Файл источников не найден: {config.SOURCES_EXCEL_PATH}'
            }), 400

        # Настройка обработчика логов для веб-интерфейса
        log_handler = WebLogHandler(current_logs)
        # Также добавляем к корневому логгеру, чтобы все логи попадали в базовый файл
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        # Создаём обработчик для полных логов в отдельный файл
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_log_filename = f"web_full_logs_{ts}.txt"
        full_log_path = f"results/{full_log_filename}"
        full_log_handler = WebFullLogHandler(full_log_path)
        root_logger.addHandler(full_log_handler)

        app_logger.info(f"Начало обработки запроса: {user_query}")

        # Получение агента и обработка запроса
        agent = get_agent()
        if not temp_excel_path:
            return jsonify({'success': False, 'error': 'Файл источников не был загружен!'}), 400
        result = agent.process_query(user_query, sources_path=temp_excel_path)

        processing_completed = True

        if result['status'] == 'success':
            app_logger.info("Запрос обработан успешно")
            # Сохраняем результат в JSON
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_filename = f"web_result_{ts}.json"
            result_path = f"results/{result_filename}"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Сохраняем отчёт в Markdown
            md_filename = f"web_report_{ts}.md"
            md_path = f"results/{md_filename}"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# Отчёт по запросу\n\n")
                f.write(f"**Запрос:** {result['user_query']}\n\n")
                f.write(f"**Дата обработки:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Статистика:**\n")
                f.write(f"- Обработано источников: {result['processed_sources']}\n")
                f.write(f"- Всего источников: {result['total_sources']}\n")
                f.write(f"- Всего документов: {result['total_documents']}\n")
                f.write(f"- Сгенерировано вопросов: {len(result['questions'])}\n\n")
                f.write(f"## Итоговый отчёт\n\n")
                f.write(result['final_report'])
                f.write(f"\n\n## Вопросы и ответы\n\n")
                for i, qa in enumerate(result['question_answers'], 1):
                    f.write(f"### Вопрос {i}\n")
                    f.write(f"{qa['question']}\n\n")
                    f.write(f"**Ответ:**\n")
                    f.write(f"{qa['answer']}\n\n")
            
            # Конвертируем Markdown в PDF
            pdf_filename = f"web_report_{ts}.pdf"
            pdf_path = f"results/{pdf_filename}"
            try:
                # Читаем markdown и конвертируем в HTML
                with open(md_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite'])
                
                # Создаём полный HTML документ
                full_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>Отчёт по запросу</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                        h2 {{ color: #34495e; margin-top: 30px; }}
                        h3 {{ color: #7f8c8d; }}
                        code {{ background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }}
                        pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                {html_content}
                </body>
                </html>
                """
                
                # Конвертируем в PDF
                font_config = FontConfiguration()
                HTML(string=full_html).write_pdf(pdf_path, font_config=font_config)
                app_logger.info(f"PDF отчёт сохранён в {pdf_path}")
            except Exception as e:
                app_logger.error(f"Ошибка при создании PDF: {e}")
                pdf_filename = None
            
            # Сохраняем логи текущего запроса
            log_filename = f"web_agent_logs_{ts}.txt"
            log_path = f"results/{log_filename}"
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(current_logs))
            app_logger.info(f"Результат сохранён в {result_path}, markdown в {md_path}, PDF в {pdf_path}, сокращённые логи в {log_path}, полные логи в {full_log_path}")
            # Добавляем ссылки на скачивание
            download_links = {
                'result': f'/results/{result_filename}',
                'markdown': f'/results/{md_filename}',
                'pdf': f'/results/{pdf_filename}' if pdf_filename else None,
                'logs': f'/results/{log_filename}',
                'full_logs': f'/results/{full_log_filename}'
            }
            return jsonify({'success': True, 'data': result, 'download_links': download_links})
        else:
            app_logger.error(f"Ошибка при обработке запроса: {result.get('error', 'Неизвестная ошибка')}")
            return jsonify({'success': False, 'error': result.get('error', 'Неизвестная ошибка')}), 500

    except Exception as e:
        app_logger.error(f"Ошибка обработки API запроса: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        # Удаляем обработчик логов
        if log_handler:
            root_logger.removeHandler(log_handler)
            log_handler = None
        if 'full_log_handler' in locals():
            root_logger.removeHandler(full_log_handler)
        # Удаляем временный файл источников
        if temp_excel_path and os.path.exists(temp_excel_path):
            try:
                os.remove(temp_excel_path)
            except Exception as e:
                app_logger.error(f"Ошибка при удалении временного файла источников: {e}")

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

@app.route('/results/<path:filename>')
def download_result(filename):
    return send_from_directory('results', filename, as_attachment=True)

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
