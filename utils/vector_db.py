
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, UpdateCollection
from datetime import datetime
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range,FilterSelector
from langchain_gigachat import GigaChatEmbeddings

import uuid
import config

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Класс для работы с векторной БД Qdrant используя GigaChat embeddings"""

    def __init__(self):
        self.client = QdrantClient(url=config.QDRANT_URL)
        self.collection_name = config.QDRANT_COLLECTION_NAME

        # Инициализируем GigaChat Embeddings с username/password авторизацией
        self.embeddings = GigaChatEmbeddings(
            user=config.GIGACHAT_USERNAME,
            password=config.GIGACHAT_PASSWORD,
            base_url=config.GIGACHAT_BASE_URL,
            auth_url=config.GIGACHAT_AUTH_URL,
            scope=config.GIGACHAT_SCOPE,
            verify_ssl_certs=config.GIGACHAT_VERIFY_SSL
        )

        # Получаем размерность векторов от первого эмбеддинга
        self.vector_size = None
        #self._setup_collection()
    def _ensure_collection(self):
        """Гарантирует существование коллекции"""
        if not self._collection_exists():
            self._setup_collection()
    def _collection_exists(self) -> bool:
        """Проверяет существование коллекции"""
        try:
            collections = self.client.get_collections()
            return any(col.name == self.collection_name 
                    for col in collections.collections)
        except Exception:
            return False
    def _get_vector_dimension(self) -> int:
        """Получает размерность векторов из GigaChat embeddings"""
        if self.vector_size is None:
            try:
                # Создаем тестовый эмбеддинг для определения размерности
                test_embedding = self.embeddings.embed_query("тест")
                self.vector_size = len(test_embedding)
                logger.info(f"Определена размерность векторов GigaChat: {self.vector_size}")
            except Exception as e:
                logger.error(f"Ошибка при определении размерности векторов: {e}")
                # Используем стандартную размерность как fallback
                self.vector_size = 1024  # Предполагаемая размерность по умолчанию
                logger.warning(f"Используется размерность по умолчанию: {self.vector_size}")

        return self.vector_size

    def _setup_collection(self):
        """Создает коллекцию в Qdrant если она не существует"""
        try:
            # Проверяем существует ли коллекция
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                vector_dim = self._get_vector_dimension()

                # Создаем коллекцию с корректной размерностью
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Создана коллекция '{self.collection_name}' с размерностью {vector_dim}")
            else:
                # Если коллекция существует, получаем информацию о размерности
                collection_info = self.client.get_collection(self.collection_name)
                existing_size = collection_info.config.params.vectors.size
                current_size = self._get_vector_dimension()

                if existing_size != current_size:
                    logger.warning(f"Размерность коллекции ({existing_size}) не совпадает с текущей ({current_size})")
                    logger.info("Удаляем и пересоздаем коллекцию с новой размерностью")

                    self.client.delete_collection(self.collection_name)
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=current_size,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Пересоздана коллекция с размерностью {current_size}")
                else:
                    logger.info(f"Коллекция '{self.collection_name}' уже существует с размерностью {existing_size}")

        except Exception as e:
            logger.error(f"Ошибка при настройке коллекции: {e}")
            raise
    def url_exists(self, url: str) -> bool:
        self._ensure_collection()
        """Проверяет, существует ли URL в базе данных"""
        logger.info((f"Проверка наличия URL в БД: {url}"))
        try:
            search_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[FieldCondition(
                        key="source_url",
                        match=MatchValue(value=url) 
                    )]
                )
            )
            logger.info((f"Найдено:{search_result.count}"))
            return search_result.count > 0
        except Exception as e:
            logger.error(f"Ошибка при проверке URL: {e}")
            return False

    def get_processing_date(self, url: str) -> Optional[str]:
        self._ensure_collection()
        """Возвращает дату обработки URL"""
        try:
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="source_url",
                        match=MatchValue(value=url)
                    )]
                ),
                limit=1
            )
            if search_result[0]:
                return search_result[0][0].payload.get('processing_date')
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении даты обработки: {e}")
            return None

    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """Добавляет документы в векторную БД"""
        self._ensure_collection()
        if not chunks:
            return
        try:
            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embeddings.embed_documents(texts)

            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                payload = {
                'content': chunk['content'],
                'source_url': chunk['source_url'],
                'chunk_index': i,
                'processing_date': datetime.now().isoformat()  # Текущая дата
                }
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
                points.append(point)

            # Загружаем точки батчами для оптимизации
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

            logger.info(f"Добавлено {len(points)} документов в векторную БД")

        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            raise

    def search_similar(self, query: str, limit: int = None, threshold: float = None) -> List[Dict]:
        """Поиск похожих документов по запросу"""
        self._ensure_collection()
        try:
            if limit is None:
                limit = 10
            if threshold is None:
                threshold = 0.9

            # Создаем эмбеддинг для запроса
            query_embedding = self.embeddings.embed_query(query)

            # Выполняем поиск
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=threshold
            )

            # Форматируем результаты
            results = []
            for scored_point in search_result:
                content = f"{scored_point.payload['content']}\n<Source>{scored_point.payload['source_url']}</Source>"
                results.append({
                    'content': content,
                    'source_url': scored_point.payload['source_url'],
                    'score': scored_point.score,
                    'id': scored_point.id,
                    'vector': scored_point.vector
                })

            logger.info(f"Найдено {len(results)} релевантных документов для запроса")
            #unique_results = self.remove_duplicates_by_vectors(results, threshold)
            unique_results=results
            logger.info(f"Из них {len(unique_results)} уникальных")            
            return unique_results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске документов: {e}")
            raise
    def remove_duplicates_by_vectors(self, search_results: List[Dict], threshold: float = None) -> List[Dict]:
        """Удаляет дубликаты из результатов поиска по сохраненным векторам"""
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD
        
        if not search_results:
            return search_results

        unique_results = []
        unique_vectors = []
        
        for result in search_results:
            vector = result['vector']  # Вектор из результатов поиска
            is_duplicate = False
            
            # Сравниваем с уже добавленными векторами
            for unique_vector in unique_vectors:
                similarity = self._cosine_similarity(vector, unique_vector)
                if similarity >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                unique_vectors.append(vector)
        
        return unique_results
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Вычисляет косинусное сходство между двумя векторами"""
        import math

        # Вычисляем скалярное произведение
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Вычисляем нормы векторов
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))

        # Избегаем деления на ноль
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def clear_collection(self):
        """Очищает коллекцию"""
        try:
            self.client.delete_collection(self.collection_name)
            self._setup_collection()
            logger.info(f"Коллекция '{self.collection_name}' очищена")
        except Exception as e:
            logger.error(f"Ошибка при очистке коллекции: {e}")
            raise
    def delete_by_date(self, max_date: str):
        """Удаляет документы, обработанные до указанной даты"""
        self._ensure_collection()
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="processing_date",
                                range=Range(lt=max_date))
                        ]
                    )
                )
            )
            logger.info(f"Удалены документы, обработанные до {max_date}")
        except Exception as e:
            logger.error(f"Ошибка при удалении по дате: {e}")
            raise
    def get_collection_info(self) -> Dict[str, Any]:
        """Возвращает информацию о коллекции"""
        self._ensure_collection()
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'vectors_count': info.vectors_count,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о коллекции: {e}")
            return {}
