import logging
import sys
from typing import List
import config

class WebLogHandler(logging.Handler):
    """Обработчик логов для веб-интерфейса (сокращённый формат)"""

    def __init__(self, logs_list: List[str]):
        super().__init__()
        self.logs_list = logs_list
        # Форматтер не нужен, web-лог — только текст сообщения

    def emit(self, record):
        """Добавляет сокращённую запись лога в список для веб-интерфейса"""
        try:
            msg = record.getMessage()
            # Пропускаем LLM-запросы и ответы
            if '[LLM REQUEST]' in msg or '[LLM RESPONSE]' in msg:
                return
            # Пропускаем логи от Flask и HTTP-библиотек
            if record.name in ['flask', 'werkzeug', 'requests', 'urllib3'] or record.name.startswith('flask') or record.name.startswith('werkzeug'):
                return
            self.logs_list.append(msg)
            # Ограничиваем количество логов для экономии памяти
            if len(self.logs_list) > 1000:
                self.logs_list.pop(0)
        except Exception:
            self.handleError(record)

class ColoredFormatter(logging.Formatter):
    """Форматтер с цветовой подсветкой для консоли"""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green  
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

def setup_logging():
    """Настройка системы логирования"""

    # Базовая настройка
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[]
    )

    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    #root_logger.setLevel(logging.DEBUG)

    # Консольный обработчик с цветами
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Файловый обработчик
    try:
        file_handler = logging.FileHandler(config.LOG_FILE_PATH, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        root_logger.error(f"Не удалось настроить файловое логирование: {e}")

    # Устанавливаем уровень для внешних библиотек
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('qdrant_client').setLevel(logging.DEBUG)

    return root_logger

def get_logger(name: str) -> logging.Logger:
    """Получает логгер с заданным именем"""
    if not logging.getLogger().handlers:
        setup_logging()

    return logging.getLogger(name)

# Инициализируем логирование при импорте модуля
setup_logging()
