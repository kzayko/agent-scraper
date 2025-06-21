
import re
from typing import List, Dict
import logging
import config

logger = logging.getLogger(__name__)

class TextProcessor:
    """Класс для обработки и разбиения текста на блоки"""

    def __init__(self):
        self.max_chunk_size = config.MAX_CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP

    def chunk_text(self, text: str, source_url: str = "", max_chunk_size: int = None) -> List[Dict[str, str]]:
        """Разбивает текст на блоки по границам параграфов и предложений"""
        if max_chunk_size is None:
            max_chunk_size = self.max_chunk_size

        if not text or not text.strip():
            return []

        # Очистка текста
        text = self._clean_text(text)

        # Разбиение на параграфы
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []

        for paragraph in paragraphs:
            if len(paragraph) <= max_chunk_size:
                # Параграф помещается в один блок
                chunks.append({
                    'content': paragraph,
                    'source_url': source_url
                })
            else:
                # Параграф слишком длинный, разбиваем по предложениям
                sentence_chunks = self._split_long_paragraph(paragraph, max_chunk_size)
                for chunk in sentence_chunks:
                    chunks.append({
                        'content': chunk,
                        'source_url': source_url
                    })

        logger.info(f"Текст разбит на {len(chunks)} блоков")
        return chunks

    def _clean_text(self, text: str) -> str:
        """Очистка текста от лишних символов и нормализация"""
        # Удаляем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Удаляем специальные символы, которые могут мешать
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        return text.strip()

    def _split_long_paragraph(self, paragraph: str, max_chunk_size: int) -> List[str]:
        """Разбивает длинный параграф на части по границам предложений"""
        # Разбиваем на предложения
        sentences = self._split_into_sentences(paragraph)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Если предложение само по себе слишком длинное
            if len(sentence) > max_chunk_size:
                # Сохраняем текущий блок, если он не пустой
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Разбиваем длинное предложение по словам
                word_chunks = self._split_by_words(sentence, max_chunk_size)
                chunks.extend(word_chunks)

            # Если добавление предложения не превысит лимит
            elif len(current_chunk + " " + sentence) <= max_chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

            # Предложение не помещается в текущий блок
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        # Добавляем последний блок
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения"""
        # Простой алгоритм разбиения на предложения
        # Ищем точки, восклицательные и вопросительные знаки
        sentence_endings = re.compile(r'[.!?]+\s+')
        sentences = sentence_endings.split(text)

        # Восстанавливаем знаки препинания
        result = []
        parts = sentence_endings.findall(text)

        for i, sentence in enumerate(sentences[:-1]):
            if i < len(parts):
                result.append(sentence + parts[i].strip())

        # Добавляем последнее предложение
        if sentences[-1].strip():
            result.append(sentences[-1].strip())

        return [s.strip() for s in result if s.strip()]

    def _split_by_words(self, text: str, max_chunk_size: int) -> List[str]:
        """Разбивает текст по словам когда предложения слишком длинные"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 для пробела

            if current_length + word_length <= max_chunk_size:
                current_chunk.append(word)
                current_length += word_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
