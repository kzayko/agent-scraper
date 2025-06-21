
import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# ========== КОНСТАНТЫ КОНФИГУРАЦИИ ==========

# Размеры и ограничения для обработки текста
MAX_CHUNK_SIZE = int(os.getenv('MAX_CHUNK_SIZE', '1000'))  # Максимальный размер блока текста
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))     # Перекрытие между блоками
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.85'))  # Порог схожести для удаления дубликатов

# Параметры векторной БД Qdrant
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'info_agent_embeddings')

# Параметры подключения к GigaChat
GIGACHAT_USERNAME = os.getenv('GIGACHAT_USERNAME')
GIGACHAT_PASSWORD = os.getenv('GIGACHAT_PASSWORD')
GIGACHAT_BASE_URL = os.getenv('GIGACHAT_BASE_URL')
GIGACHAT_AUTH_URL = os.getenv('GIGACHAT_AUTH_URL')
GIGACHAT_SCOPE = os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS')
GIGACHAT_VERIFY_SSL = os.getenv('GIGACHAT_VERIFY_SSL', 'False').lower() == 'true'

# Путь к Excel файлу с источниками
SOURCES_EXCEL_PATH = os.getenv('SOURCES_EXCEL_PATH', 'sources.xlsx')

# Параметры логирования
LOG_FILE_PATH = os.getenv('LOG_FILE_PATH', 'agent_logs.txt')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Параметры веб-интерфейса
WEB_HOST = os.getenv('WEB_HOST', '0.0.0.0')
WEB_PORT = int(os.getenv('WEB_PORT', '5000'))
WEB_DEBUG = os.getenv('WEB_DEBUG', 'False').lower() == 'true'

# Валидация обязательных параметров
if not GIGACHAT_USERNAME or not GIGACHAT_PASSWORD:
    raise ValueError("GIGACHAT_USERNAME и GIGACHAT_PASSWORD должны быть установлены в переменных окружения")
