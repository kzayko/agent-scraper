import logging
from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage
import pandas as pd
import json
import asyncio
import config
from utils.logger import get_logger
from utils.vector_db import VectorDatabase
from utils.web_parser import WebParser
from utils.text_processor import TextProcessor
import re

# Настройка логирования
agent_logger = get_logger("agent")

class AgentState(TypedDict):
    """Состояние агента для LangGraph"""
    user_query: str
    questions: List[str]
    sources: List[str]
    processed_sources: int
    documents: int
    question_answers: List[Dict[str, str]]
    final_report: str
    current_step: str
    error: str

class InformationSummarizerAgent:
    """Агент-суммаризатор информации с использованием LangGraph и GigaChat"""

    def __init__(self):
        # Инициализация GigaChat для LLM операций с username/password авторизацией
        self.llm = GigaChat(
            user=config.GIGACHAT_USERNAME,
            password=config.GIGACHAT_PASSWORD,
            base_url=config.GIGACHAT_BASE_URL,
            auth_url=config.GIGACHAT_AUTH_URL,
            scope=config.GIGACHAT_SCOPE,
            verify_ssl_certs=config.GIGACHAT_VERIFY_SSL,
            profanity_check=config.GIGACHAT_PROFANITY_CHECK,
	    temperature=0.1
        )

        # Инициализация вспомогательных модулей
        self.vector_db = VectorDatabase()
        self.web_parser = WebParser()
        self.text_processor = TextProcessor()

        # Граф без sources_path по умолчанию (для CLI)
        self.graph = self._create_graph()

        agent_logger.info("Агент-суммаризатор инициализирован с GigaChat")

    def _create_graph(self, sources_path: str = None) -> StateGraph:
        """Создает граф состояний для агента"""
        workflow = StateGraph(AgentState)

        # Добавляем узлы
        workflow.add_node("generate_questions", self._generate_questions)
        if sources_path is not None:
            workflow.add_node("load_sources", lambda state: self._load_sources(state, sources_path))
        else:
            workflow.add_node("load_sources", self._load_sources)
        workflow.add_node("process_sources", self._process_sources)
        workflow.add_node("answer_questions", self._answer_questions)
        workflow.add_node("generate_report", self._generate_report)

        # Добавляем рёбра
        workflow.add_edge(START, "generate_questions")
        workflow.add_edge("generate_questions", "load_sources")
        workflow.add_edge("load_sources", "process_sources")
        workflow.add_edge("process_sources", "answer_questions")
        workflow.add_edge("answer_questions", "generate_report")
        workflow.add_edge("generate_report", END)

        compiled_graph = workflow.compile()
        return compiled_graph  # type: ignore

    def _generate_questions(self, state: AgentState) -> AgentState:
        """Генерирует вопросы на основе пользовательского запроса"""
        try:
            state["current_step"] = "Генерация вопросов"
            agent_logger.info(f"Шаг 1: {state['current_step']}")

            prompt = f"""
            Задача: Разбить пользовательский запрос на смысловые блоки и сформулировать вопросы к каждому блоку.

            Пользовательский запрос: "{state['user_query']}"

            Проанализируй запрос и:
            1. Определи основные смысловые блоки
            2. Сформулируй конкретный вопрос для каждого блока
            3. Убедись, что вопросы покрывают весь запрос

            Верни ответ строго в JSON формате:
            {{
                "questions": [
                    "Вопрос 1",
                    "Вопрос 2",
                    "Вопрос 3"
                ]
            }}
            """

            message = HumanMessage(content=prompt)
            agent_logger.info(f"[LLM REQUEST] PROMPT: {prompt.strip()}")
            response = self.llm.invoke([message])
            agent_logger.info(f"[LLM RESPONSE] RESPONSE: {response.content.strip()}")

            # Парсим JSON ответ
            try:
                clean_content = self.clean_json_str(response.content)
                result = json.loads(clean_content)
                questions = result.get("questions", [])

                if not questions:
                    # Fallback: создаем один вопрос из исходного запроса
                    questions = [state['user_query']]

                state["questions"] = questions
                agent_logger.info(f"Сгенерировано {len(questions)} вопросов")

                for i, question in enumerate(questions, 1):
                    agent_logger.info(f"Вопрос {i}: {question}")

            except json.JSONDecodeError as e:
                agent_logger.error(f"Ошибка парсинга JSON ответа: {e}")
                # Fallback: используем исходный запрос
                state["questions"] = [state['user_query']]

        except Exception as e:
            agent_logger.error(f"Ошибка при генерации вопросов: {e}")
            state["error"] = str(e)
            state["questions"] = [state['user_query']]  # Fallback

        return state

    def clean_json_str(self, s: str) -> str:
        """Удаляет markdown-блоки (```), лишние кавычки и пробелы для корректного парсинга JSON."""
        s = s.strip()
        # Удалить markdown-блоки
        s = re.sub(r'^```[a-zA-Z]*', '', s)
        s = re.sub(r'```$', '', s)
        s = s.strip()
        # Удалить одиночные/двойные кавычки вокруг всего блока
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        return s

    def normalize_url(self, url: str) -> str:
        return url.strip().rstrip('/').lower()

    def _load_sources(self, state: AgentState, sources_path: str = None) -> AgentState:
        """Загружает список источников из Excel файла"""
        try:
            state["current_step"] = "Загрузка источников"
            agent_logger.info(f"Шаг 2: {state['current_step']}")

            # Для WEB-режима sources_path обязателен, для CLI можно использовать config
            if sources_path is None:
                import config
                excel_path = getattr(config, 'SOURCES_EXCEL_PATH', None)
                if not excel_path:
                    raise ValueError("Путь к файлу источников не задан!")
            else:
                excel_path = sources_path
            df = pd.read_excel(excel_path)

            # Предполагаем, что URL находятся в первой колонке
            if 'url' in df.columns:
                urls = df['url'].dropna().tolist()
            elif 'URL' in df.columns:
                urls = df['URL'].dropna().tolist()
            else:
                # Берем первую колонку
                urls = df.iloc[:, 0].dropna().tolist()

            # Нормализуем все URL
            state["sources"] = [self.normalize_url(u) for u in urls]
            state["processed_sources"] = 0

            agent_logger.info(f"Список источников (state['sources']): {state['sources']}")
            agent_logger.info(f"Загружено {len(urls)} источников из {excel_path}")

        except Exception as e:
            agent_logger.error(f"Ошибка при загрузке источников: {e}")
            state["error"] = str(e)
            state["sources"] = []

        return state

    def _process_sources(self, state: AgentState) -> AgentState:
        """Обрабатывает каждый источник: парсинг, разбиение на блоки, создание эмбеддингов"""
        try:
            state["current_step"] = "Обработка источников"
            agent_logger.info(f"Шаг 3: {state['current_step']}")
            total_documents = 0  # Счетчик общего количества документов

            # Обрабатываем только те источники, которые есть в текущем списке state["sources"]
            current_sources = list(state["sources"])
            for i, url in enumerate(current_sources, 1):
                try:
                    agent_logger.info(f"Обработка источника {i}/{len(state['sources'])}: {url}")
                    
                    # Проверяем существование URL в БД
                    if self.vector_db.url_exists(url):
                        processing_date = self.vector_db.get_processing_date(url)
                        agent_logger.info(f"Ссылка уже обработана: {url} (дата обработки: {processing_date})")
                        continue
                    
                    # Парсим веб-страницу
                    content = self.web_parser.parse_url(url)
                    if not content:
                        agent_logger.warning(f"Не удалось извлечь контент из {url}")
                        continue
                    
                    # Разбиваем текст на блоки
                    chunks = self.text_processor.chunk_text(content, url)
                    if not chunks:
                        agent_logger.warning(f"Не удалось разбить контент из {url} на блоки")
                        continue
                    
                    # Немедленно добавляем чанки в векторную БД
                    agent_logger.info(f"Добавление {len(chunks)} блоков из источника {url} в векторную БД")
                    self.vector_db.add_documents(chunks)
                    total_documents += len(chunks)
                    state["processed_sources"] += 1
                    agent_logger.info(f"Источник {url} обработан, добавлено {len(chunks)} блоков")
                    
                except Exception as e:
                    agent_logger.error(f"Ошибка при обработке источника {url}: {e}")
                    # Ошибка была в том, что после блока except не было обработки ошибки в state.
                    # Исправлено: теперь при ошибке в обработке источника, ошибка логируется и добавляется в state["error_details"].
                    if "error_details" not in state:
                        state["error_details"] = []
                    state["error_details"].append({"url": url, "error": str(e)})
                    continue

            state["documents"] = total_documents  # Сохраняем общее количество документов
            agent_logger.info(f"Обработано {state['processed_sources']} источников, всего документов: {total_documents}")

        except Exception as e:
            agent_logger.error(f"Ошибка при обработке источников: {e}")
            state["error"] = str(e)
            state["documents"] = 0
            state["error"] = str(e)
            state["documents"] = 0

        return state


    def _answer_questions(self, state: AgentState) -> AgentState:
        """Отвечает на каждый вопрос используя релевантные блоки из векторной БД"""
        try:
            state["current_step"] = "Ответы на вопросы"
            agent_logger.info(f"Шаг 4: {state['current_step']}")

            question_answers = []

            for i, question in enumerate(state["questions"], 1):
                try:
                    agent_logger.info(f"Обработка вопроса {i}/{len(state['questions'])}: {question}")

                    # Ищем релевантные документы
                    relevant_docs = self.vector_db.search_similar(
                        query=question,
                        limit=config.DOCS_PER_ANSWER,
                        threshold=config.SIMILARITY_THRESHOLD
                    )

                    # Фильтрация: только параграфы из разрешённых источников (загруженных изначально)
                    allowed_sources = set(state["sources"])
                    relevant_docs = [doc for doc in relevant_docs if self.normalize_url(doc.get("source_url", "")) in allowed_sources]

                    if not relevant_docs:
                        agent_logger.warning(f"Не найдено релевантных документов для вопроса: {question}")
                        question_answers.append({
                            "question": question,
                            "answer": "К сожалению, не найдено релевантной информации для ответа на этот вопрос."
                        })
                        continue

                    # Удаляем дубликаты (не нужно пока) 
                    unique_docs=relevant_docs
                    # unique_docs = self.vector_db.remove_duplicates(relevant_docs)
                    # agent_logger.info(f"Найдено {len(unique_docs)} уникальных релевантных документов")

                    # Объединяем контент документов
                    combined_content = "\n\n".join([doc["content"] for doc in unique_docs])

                    # Формируем промпт для ответа на вопрос
                    prompt = f"""
                    Задача: Подготовь детальный ответ на вопрос, используя предоставленную информацию.

                    Вопрос: {question}

                    Доступная информация из источников:
                    {combined_content}

                    Требования к ответу:
                    1. Ответ должен быть полным и информативным
                    2. Основывайся только на предоставленной информации
                    3. Если информации недостаточно, укажи это
                    4. Структурируй ответ логично
                    5. Используй конкретные факты и данные из источников
                    6. Источник каждого абзаца указан в теге <Source>..</Source>. В тексте всегда указывай в скобках ссылку на источник, из которого он взят.

                    Ответ:
                    """

                    message = HumanMessage(content=prompt)
                    agent_logger.info(f"[LLM REQUEST] PROMPT: {prompt.strip()}")
                    response = self.llm.invoke([message])
                    agent_logger.info(f"[LLM RESPONSE] RESPONSE: {response.content.strip()}")

                    question_answers.append({
                        "question": question,
                        "answer": response.content
                    })

                    agent_logger.info(f"Сгенерирован ответ на вопрос {i}")

                except Exception as e:
                    agent_logger.error(f"Ошибка при обработке вопроса '{question}': {e}")
                    question_answers.append({
                        "question": question,
                        "answer": f"Ошибка при генерации ответа: {str(e)}"
                    })

            state["question_answers"] = question_answers
            agent_logger.info(f"Сгенерированы ответы на {len(question_answers)} вопросов")

        except Exception as e:
            agent_logger.error(f"Ошибка при генерации ответов: {e}")
            state["error"] = str(e)
            state["question_answers"] = []

        return state

    def _generate_report(self, state: AgentState) -> AgentState:
        """Генерирует итоговый отчет"""
        try:
            state["current_step"] = "Генерация итогового отчета"
            agent_logger.info(f"Шаг 5: {state['current_step']}")

            # Формируем текст с вопросами и ответами
            qa_text = ""
            for i, qa in enumerate(state["question_answers"], 1):
                qa_text += f"\n\nВопрос {i}: {qa['question']}\n"
                qa_text += f"Ответ {i}: {qa['answer']}"

            prompt = f"""
            Задача: Подготовь связный итоговый отчет на основе изначального запроса пользователя и полученных ответов на вопросы.

            Изначальный запрос пользователя: "{state['user_query']}"

            Вопросы и ответы:
            {qa_text}

            Требования к отчету:
            1. Отчет должен полностью отвечать на изначальный запрос пользователя
            2. Объедини информацию из всех ответов в логическую структуру
            3. Используй заголовки и подзаголовки для структурирования
            4. Включи конкретные факты и данные
            5. Сделай заключение с основными выводами
            6. Отчет должен быть полным и информативным
            7. Приведи ссылки на источники в конце отчета

            Итоговый отчет:
            """

            message = HumanMessage(content=prompt)
            agent_logger.info(f"[LLM REQUEST] PROMPT: {prompt.strip()}")
            response = self.llm.invoke([message])
            agent_logger.info(f"[LLM RESPONSE] RESPONSE: {response.content.strip()}")

            state["final_report"] = response.content
            state["current_step"] = "Завершено"

            agent_logger.info("Итоговый отчет сгенерирован успешно")

        except Exception as e:
            agent_logger.error(f"Ошибка при генерации отчета: {e}")
            state["error"] = str(e)
            state["final_report"] = "Ошибка при генерации итогового отчета"

        return state

    def process_query(self, user_query: str, sources_path: str) -> Dict[str, Any]:
        """Основной метод обработки запроса пользователя"""
        try:
            agent_logger.info(f"Начало обработки запроса: {user_query}")

            # Инициализируем состояние
            initial_state = AgentState(
                user_query=user_query,
                questions=[],
                sources=[],
                processed_sources=0,
                documents=0,
                question_answers=[],
                final_report="",
                current_step="Инициализация",
                error=""
            )

            # Создаём новый граф для каждого запроса
            graph = self._create_graph(sources_path)

            final_state = graph.invoke(initial_state)

            # Формируем результат
            result = {
                "user_query": final_state["user_query"],
                "questions": final_state["questions"],
                "processed_sources": final_state["processed_sources"],
                "total_sources": len(final_state["sources"]),
                "total_documents": final_state["documents"],
                "question_answers": final_state["question_answers"],
                "final_report": final_state["final_report"],
                "status": "success" if not final_state.get("error") else "error",
                "error": final_state.get("error", "")
            }

            agent_logger.info("Обработка запроса завершена успешно")
            return result

        except Exception as e:
            agent_logger.error(f"Критическая ошибка при обработке запроса: {e}")
            return {
                "user_query": user_query,
                "status": "error",
                "error": str(e),
                "final_report": "Произошла ошибка при обработке запроса"
            }

def get_agent() -> InformationSummarizerAgent:
    """Функция для получения экземпляра агента"""
    return InformationSummarizerAgent()
