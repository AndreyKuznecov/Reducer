# Reducer
Reducer wieght prediction

Входный параметры задаются в файле data\input.csv
Выходные значения находятся в файле data\output.csv

Запуск проекта:
 - установить docker
 - создать образ командой: docker build --tag reducer:1.0 .
 - запустить контейнер
 - скопировать файл с результатами из директории контейнера в директорию проекта командой: docker cp reducer:app/data/output.csv ./data