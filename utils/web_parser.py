
import requests
from bs4 import BeautifulSoup
import logging
from typing import Optional
import time
import re

logger = logging.getLogger(__name__)

class WebParser:
    """Класс для парсинга веб-страниц"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.timeout = 30
        self.max_retries = 3

    def parse_url(self, url: str) -> Optional[str]:
        """Парсит URL и возвращает текстовый контент"""
        try:
            from utils.vector_db import VectorDatabase
            vector_db = VectorDatabase()
    
            if vector_db.url_exists(url):
                processing_date = vector_db.get_processing_date(url)
                logger.info(f"Ссылка уже обработана: {url} (дата обработки: {processing_date})")
                return None  # Пропускаем обработку
            logger.info(f"Парсинг URL: {url}")

            for attempt in range(self.max_retries):
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    response.raise_for_status()

                    # Определяем кодировку
                    if response.encoding == 'ISO-8859-1' and 'charset' not in response.headers.get('content-type', ''):
                        response.encoding = response.apparent_encoding

                    content = self._extract_text_content(response.text)

                    if content:
                        logger.info(f"Успешно извлечен контент из {url} ({len(content)} символов)")
                        return content
                    else:
                        logger.warning(f"Не удалось извлечь текстовый контент из {url}")
                        return None

                except requests.exceptions.RequestException as e:
                    logger.warning(f"Попытка {attempt + 1}/{self.max_retries} не удалась для {url}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Экспоненциальная задержка
                    continue

            logger.error(f"Не удалось загрузить {url} после {self.max_retries} попыток")
            return None

        except Exception as e:
            logger.error(f"Ошибка при парсинге {url}: {e}")
            return None

    def _extract_text_content(self, html: str) -> Optional[str]:
        """Извлекает текстовый контент из HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Удаляем скрипты, стили и другие нетекстовые элементы
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'menu']):
                element.decompose()

            # Извлекаем текст
            text = soup.get_text()

            # Очистка текста
            text = self._clean_extracted_text(text)

            return text if len(text.strip()) > 100 else None  # Минимальная длина контента

        except Exception as e:
            logger.error(f"Ошибка при извлечении текста из HTML: {e}")
            return None

    def _clean_extracted_text(self, text: str) -> str:
        """Очистка извлеченного текста"""
        # Удаляем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Удаляем короткие строки (вероятно, навигация или реклама)
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Пропускаем короткие строки и строки только с символами
            if len(line) > 20 and not re.match(r'^[^a-zA-Zа-яА-Я]*$', line):
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)
