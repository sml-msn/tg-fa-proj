# Tg-Fa transliteration project.

Приложение для двусторонней (таджикско-персидской и персидско-таджикской) транслитерации.

### Состав команды и роли участников в создании проекта:
* [Мусин Шамиль](https://github.com/sml-msn) – тимлид, развертывание приложения на платформе Yandex.Cloud.
* [Красильникова Юлия](https://github.com/Jul-Kras) – написание тестов.
* [Петухова Ольга](https://github.com/petuxovao00) – создание демо-режима приложения.

### Алгоритм работы приложения:
1) Ввод текста пользователем на таджикском/персидском языке.
2) В зависимости от типа введенного пользователем текста, возможны следующие варианты работы приложения:
    *	введенный текст состоит из предложений – происходит разбиение текста на отдельные предложения с последующей их транслитерацией;
	  * введенный текст представлен одним предложением – транслитерация происходит сразу.
3) Транслитерация введенного пользователем текста осуществляется с помощью модели, представленной на платформе Hugging Face, [pst5-tg-fa-bidirectional](https://huggingface.co/sml-msn/pst5-tg-fa-bidirectional).
4) Для удобства пользователя приложение включает в себя демо-режим, благодаря которому можно протестировать работу приложения. Работа в демо-режиме происходит следующим образом:
	  * Одна строка из тестового датасета test.csv транслитерируется с таджикского языка на персидский и наоборот.
	  * Для демонстрации точности транслитерации на экран выводятся следующие метрики:
- Levenshtein distance – расстояние Левенштейна – минимальное число односимвольных преобразований (удалений, вставок или замен), необходимых для превращения одной последовательности в другую.

- Levenshtein ratio – коэффициент Левенштейна = (колич. символов-min⁡ колич. исправлений)/(колич. символов).
