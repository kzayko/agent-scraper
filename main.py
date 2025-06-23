#!/usr/bin/env python3
"""
Основной модуль для запуска агента-суммаризатора информации
"""

import argparse
from datetime import datetime  # Исправлено: импортируем datetime здесь
import sys
import json
import os
from pathlib import Path

def main():
    """Основная функция запуска"""
    parser = argparse.ArgumentParser(
        description='Агент-суммаризатор информации с GigaChat и Qdrant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --web                                    # Запуск веб-интерфейса
  python main.py "Расскажи о машинном обучении"           # Обработка запроса
  python main.py "Анализ рынка" --output result.json     # Сохранение в файл
  python main.py --health                                # Проверка состояния
  python main.py --clear-db                                # Очистка БД
        """
    )

    parser.add_argument(
        'query', 
        nargs='?', 
        help='Запрос пользователя для обработки'
    )

    parser.add_argument(
        '--web', 
        action='store_true',
        help='Запустить веб-интерфейс'
    )

    parser.add_argument(
        '--output', '-o',
        help='Путь для сохранения результата (JSON файл)'
    )

    parser.add_argument(
        '--sources', '-s',
        default='sources.xlsx',
        help='Путь к Excel файлу с источниками (по умолчанию: sources.xlsx)'
    )

    parser.add_argument(
        '--health',
        action='store_true',
        help='Проверить состояние системы'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод'
    )

    parser.add_argument(
 	'--clear-db',
    	action='store_true',
    	help='Полностью очистить векторную БД'
    )

    parser.add_argument(
    	'--clear-before-date',
    	help='Удалить документы, обработанные до указанной даты (формат: YYYY-MM-DD)'
    )
    args = parser.parse_args()

    # Настройка логирования
    if args.verbose:
      import logging
      logging.getLogger().setLevel(logging.DEBUG)
    # Очистка БД
    if args.clear_db:
      from utils.vector_db import VectorDatabase
      vector_db = VectorDatabase()
      vector_db.clear_collection()
      print("🗑️ Векторная БД полностью очищена")
      sys.exit(0)
    # Очистка БД от старых документов
    if args.clear_before_date:
      from utils.vector_db import VectorDatabase
      vector_db = VectorDatabase()
      vector_db.delete_by_date(args.clear_before_date)
      print(f"🗑️ Удалены документы, обработанные до {args.clear_before_date}")
      sys.exit(0)
    try:
        if args.health:
            run_health_check()
        elif args.web:
            run_web_interface()
        elif args.query:
            run_query_processing(args.query, args.sources, args.output)
        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n⏹️  Остановлено пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

def run_health_check():
    """Проверка состояния системы"""
    print("🔍 Проверка состояния системы...")

    try:
        import config
        print(f"✅ Конфигурация загружена")
        print(f"   - Qdrant URL: {config.QDRANT_URL}")
        print(f"   - GigaChat Username: {config.GIGACHAT_USERNAME}")
        print(f"   - Коллекция: {config.QDRANT_COLLECTION_NAME}")

        # Проверка GigaChat
        print("\n🤖 Проверка подключения к GigaChat...")
        from langchain_gigachat import GigaChat
        llm = GigaChat(
            user=config.GIGACHAT_USERNAME,
            password=config.GIGACHAT_PASSWORD,
            verify_ssl_certs=config.GIGACHAT_VERIFY_SSL,
            scope=config.GIGACHAT_SCOPE,
            profanity_check=config.GIGACHAT_PROFANITY_CHECK
        )

        test_response = llm.invoke("Привет")
        print(f"✅ GigaChat доступен (ответ: {test_response.content[:50]}...)")

        # Проверка Qdrant
        print("\n🗃️  Проверка подключения к Qdrant...")
        from utils.vector_db import VectorDatabase
        vector_db = VectorDatabase()
        info = vector_db.get_collection_info()
        print(f"✅ Qdrant доступен")
        print(f"   - Коллекция: {info.get('name', 'N/A')}")
        print(f"   - Векторов: {info.get('vectors_count', 0)}")
        print(f"   - Размерность: {info.get('vector_size', 'N/A')}")

        # Проверка файла источников
        print("\n📄 Проверка файла источников...")
        if os.path.exists(config.SOURCES_EXCEL_PATH):
            import pandas as pd
            df = pd.read_excel(config.SOURCES_EXCEL_PATH)
            print(f"✅ Файл источников найден ({len(df)} записей)")
        else:
            print(f"⚠️  Файл источников не найден: {config.SOURCES_EXCEL_PATH}")

        print("\n✅ Система готова к работе!")

    except Exception as e:
        print(f"❌ Ошибка при проверке: {e}")
        sys.exit(1)

def run_web_interface():
    """Запуск веб-интерфейса"""
    print("🌐 Запуск веб-интерфейса...")

    try:
        from app import run_web_app
        run_web_app()
    except ImportError as e:
        print(f"❌ Ошибка импорта веб-приложения: {e}")
        sys.exit(1)

def run_query_processing(query: str, sources_file: str, output_file: str):
    """Обработка запроса пользователя"""
    print(f"🚀 Обработка запроса: {query}")
    print(f"📁 Источники: {sources_file}")

    try:
        # Проверяем существование файла источников
        if not os.path.exists(sources_file):
            print(f"❌ Файл источников не найден: {sources_file}")
            sys.exit(1)

        # Устанавливаем путь к источникам
        import config
        config.SOURCES_EXCEL_PATH = sources_file

        # Получаем агента и обрабатываем запрос
        from agent import get_agent
        agent = get_agent()

        print("⏳ Обработка запроса...")
        result = agent.process_query(query)

        if result['status'] == 'success':
            print("\n✅ Обработка завершена успешно!")
            print(f"📊 Статистика:")
            print(f"   - Обработано источников: {result['processed_sources']}")
            print(f"   - Всего документов: {result['total_documents']}")
            print(f"   - Сгенерировано вопросов: {len(result['questions'])}")

            print(f"\n📝 Итоговый отчет:")
            print("=" * 80)
            print(result['final_report'])
            print("=" * 80)

            # Сохраняем результат всегда
            if args.output:
                output_path = args.output
            else:
                os.makedirs('results', exist_ok=True)
                output_path = f"results/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Результат сохранён в {output_path}")

        else:
            print(f"❌ Ошибка при обработке: {result.get('error', 'Неизвестная ошибка')}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Ошибка при обработке запроса: {e}")
        sys.exit(1)

def save_result_to_file(result: dict, output_file: str):
    """Сохранение результата в файл"""
    try:
        # Создаем директорию если её нет
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем в JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"⚠️  Ошибка при сохранении в файл: {e}")

if __name__ == '__main__':
    main()
