"""
Данный файл предназначен для очистки и трансформации данных редукторов, извлеченных из pdf-каталога
"""

import os
from pathlib import Path
import tabula as tb
import pandas as pd
import numpy as np
import sqlalchemy as sql
import warnings


def is_number(str):
    try:
        if pd.isnull(str):
            return False
        float(str)
        return True
    except ValueError:
        return False


def displace(df):
    """
    Функция выравнивает смещенные столбцы
    """
    a = df.copy()

    # извлекаем смещенные строки
    mask = a[0].apply(is_number)
    pd_disp = df[mask]

    # выравнивание столбцов
    n = len(pd_disp.columns)
    for i in range(n - 1, 0, -1):
        pd_disp.iloc[:, i] = pd_disp.iloc[:, i - 1]
    pd_disp.iloc[:, 0] = np.nan
    a[mask] = pd_disp

    return a


def merge_tables(first_table=None, second_table=None):
    """
    Функция извлекает значения массы из второй таблицы
    и добавляет их к первой
    """
    df1 = first_table.copy()
    df2 = second_table.copy()

    # проебразование второй таблицы, извлечение индекса и типа редуктора
    n2 = len(df2.columns)
    df2.replace(',', '.', regex=True, inplace=True)
    df2.drop([0, 1], inplace=True)
    df2.drop([i for i in range(1, n2 - 1)], axis=1, inplace=True)
    df2[0] = df2[0].str.findall(r'[A-Z]{2,3}\s?\d{2,3}')
    df2 = df2.explode(0).copy()
    df2[0] = df2[0].str.replace(r'\s', '')

    # проебразование первой таблицы, извлечение индекса и типа редуктора
    df1[0] = df1[0].str.findall(r'[A-Z]{2,3}\s?\d{2,3}')
    df1 = df1.explode(0, ignore_index=True).copy()
    df1[0] = df1[0].str.replace(r'\s', '')

    df1 = df1.merge(right=df2, left_on=0, right_on=0, how='inner', validate='m:1', suffixes=('_left', '_right'))

    return df1


def pdf_table1(pages=None, columns=None):
    """
    Функция извлекает таблицы из указанных страниц pdf-файла
    """
    df = pd.DataFrame(columns=columns)
    pages_valid = list(range(98, 111, 2)) + list(range(246, 253, 2))
    pages_set1 = list(range(98, 111, 2))
    pages_set2 = list(range(246, 253, 2))
    col_float = [1, 2, 3, 4, 5, 6, 7, 8, 11]
    for page in pages:
        # проверка на допустимость номера страницы
        if page not in pages_valid:
            raise ValueError(f"Недопустиный номер страницы: {page}")

        file = tb.read_pdf(cat_path, pages=page, lattice=True, pandas_options={'header': None})

        file[0].replace(',', '.', regex=True, inplace=True)
        file[0].drop([0, 1, 2], inplace=True)
        file[0].reset_index(drop=True, inplace=True)

        # выравнивание таблиц обоих сетов и добавление значений массы в таблицу для второго сета
        if page in pages_set1:
            # выравнивание строк
            file[0] = displace(file[0])
        elif page in pages_set2:
            # выравнивание строк
            file[0].iloc[1:] = displace(file[0].iloc[1:])
            file[0][11] = file[1].iloc[2, 7].replace(',', '.')

            # замена неизвестных символов на NaN
        mask = file[0][col_float].applymap(is_number)
        df_temp = file[0][col_float].copy()
        df_temp[~mask] = np.nan
        file[0][col_float] = df_temp

        if page in pages_set1:
            col_drop_excp = [9, 10, 12, 13, 14, 15]
            col_drop = [9, 10, 12, 13, 14, 15, 16]

            # заполнение столбца с массой
            file[0][11].fillna(method='ffill', inplace=True)

            # заполнение столбца с типом редуктора
            file[0][0].fillna(method='ffill', inplace=True)
            file[0][0] = file[0][0].str.findall(r'^[A-Z]{2,3}')
            file[0] = file[0].explode(0).copy()
        elif page in pages_set2:
            col_drop_excp = [9, 10, 12, 13]
            col_drop = [9, 10, 12, 13, 14]
            file[0][0] = 'SRO'

        if page == 98 or page == 246:
            col_dr = col_drop_excp
        else:
            col_dr = col_drop
        file[0].drop(columns=col_dr, inplace=True)

        file[0][col_float] = file[0][col_float].astype(float)

        file[0].columns = columns
        df = pd.concat([df, file[0]], ignore_index=True)
    return df


def pdf_table2(pages=None, columns=None):
    """
    Функция извлекает таблицы из указанных страниц pdf-файла
    """
    df = pd.DataFrame(columns=columns)
    pages_valid = list(range(138, 149, 2)) + list(range(184, 195, 2))
    col_drop = [9, 10, 11, 12, 13]
    col_float = [1, 2, 3, 4, 5, 6, 7, 8, 14]
    for page in pages:
        # проверка на допустимость номера страницы
        if page not in pages_valid:
            raise ValueError(f"Недопустиный номер страницы: {page}")

        file = tb.read_pdf(cat_path, pages=page, lattice=True, pandas_options={'header': None})

        file[0].replace(',', '.', regex=True, inplace=True)
        file[0].drop([0, 1, 2], inplace=True)
        file[0].reset_index(drop=True, inplace=True)

        # выравнивание строк
        file[0] = displace(file[0])

        # заполнение столбца с типом редуктора
        file[0][0].fillna(method='ffill', inplace=True)
        if page == 138:
            file[0][0] = 'F' + file[0][0]  # исправление опечатки в каталоге

        # добавление столбца с массой к текущей таблице
        file[0] = merge_tables(first_table=file[0], second_table=file[1])
        file[0].rename({'7_left': 7, '7_right': 14}, axis=1, inplace=True)

        # замена неизвестных символов на NaN
        mask = file[0][col_float].applymap(is_number)
        df_temp = file[0][col_float].copy()
        df_temp[~mask] = np.nan
        file[0][col_float] = df_temp

        # заполнение столбца с типом редуктора
        file[0][0] = file[0][0].str.findall(r'[A-Z]{2,3}')
        file[0] = file[0].explode(0).copy()

        file[0].drop(columns=col_drop, inplace=True)

        file[0][col_float] = file[0][col_float].astype(float)

        file[0].columns = columns
        df = pd.concat([df, file[0]], ignore_index=True)
    return df


warnings.filterwarnings("ignore")

try:
    current_path = Path(os.path.abspath('parsing_pdf.ipynb'))
    cat_path = Path(os.path.join(current_path.parents[0], 'catalog', 'Каталог_механика_полный.pdf'))
    if not os.path.isfile(cat_path):
        raise FileNotFoundError('Файл не найден')
    pages1 = [98, 100, 102, 104, 106, 108, 110]
    pages2 = [138, 140, 142, 144, 146, 148]
    pages3 = [184, 186, 188, 190, 192, 194]
    pages4 = [246, 248, 250, 252]
    columns = ['type', 'i', 'i_f', 'n2', 'M2', 'P1', 'Fr1', 'Fr2', 'J1', 'm']

    data1 = pdf_table1(pages=pages1 + pages4, columns=columns)
    data2 = pdf_table2(pages=pages2 + pages3, columns=columns)
    data = pd.concat([data1, data2], ignore_index=True)

    engine = sql.create_engine('mysql://root:3456@127.0.0.1:3306/reducer_db')
    with engine.connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS reducers(
                        id SMALLINT UNSIGNED NOT NULL PRIMARY KEY AUTO_INCREMENT,
                        type VARCHAR(6) NULL,
                        i FLOAT NULL,
                        i_f FLOAT NULL,
                        n2 FLOAT NULL,
                        M2 FLOAT NULL,
                        P1 FLOAT NULL,
                        Fr1 FLOAT NULL,
                        Fr2 FLOAT NULL,
                        J1 FLOAT NULL,
                        m FLOAT NULL,
                        CONSTRAINT U_K UNIQUE (type, i, M2, P1, m)
                           )""")
        try:
            data.to_sql('reducers', conn, index=False, if_exists='append')
        except sql.exc.IntegrityError as e:
            # предотваращаем увеличение значений в столбце ID(AUTO_INCREMENT) при неудачных выгрузках
            conn.execute("ALTER TABLE reducers AUTO_INCREMENT = 1")
            print("Вставка дублирующихся значений " + str(e))

except Exception as e:
    print(e)
