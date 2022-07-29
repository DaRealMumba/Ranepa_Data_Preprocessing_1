from math import remainder
from nbformat import write
import streamlit as st
import pandas as pd 
import numpy as np 
from PIL import Image
from pymystem3 import Mystem

# задаем функцию, к которой будем обращаться ниже
m = Mystem()
def lemmatization (purpose):
  lemma_tmp=m.lemmatize(purpose)
          
  if ('жилье' in lemma_tmp) or ('недвижимость' in lemma_tmp):
      lemma_name_purpose='жилье'
      return lemma_name_purpose
  elif 'автомобиль' in lemma_tmp:
      lemma_name_purpose='автомобиль'
      return lemma_name_purpose
  elif 'образование' in lemma_tmp:
      lemma_name_purpose='образование'
      return lemma_name_purpose
  else:
      lemma_name_purpose='личные цели'
      return lemma_name_purpose


st.markdown('''<h1 style='text-align: center; color: black;'
            >Предварительная обработка данных часть 1: Знакомство с инструментами</h1>''', 
            unsafe_allow_html=True)

st.write("""
Предварительная обработка данных (Data Preprocessing) - важный этап в науке о данных. Для того, чтобы обучать модель, ей необходимо предоставить очищенные данные в понятном ей виде. 
Предварительная обработка данных, наравне с разведочным анализом данных, обычно занимает большую часть проекта.
\n **Полезно почитать:** [1](https://ru.wikipedia.org/wiki/Предварительная_обработка_данных), [2](https://habr.com/ru/post/511132/), [3](https://pythobyte.com/data-preprocessing-0cb9135c/)

\nЛабораторная работа "Предварительная обработка данных" состоит из 2 частей:
\n**Первая часть:** посмотрим, какие есть способы и возможности обработки данных.
\n**Вторая часть:** обработаем одни и те же данные двумя разными способами, обучим модель и сравним результаты (**[ссылка](https://darealmumba-ranepa-data-preprocessi-data-preprocessing-2-j07j4m.streamlitapp.com/)**). 
""")


image = Image.open('images/Pipeline_1.png')
st.image(image)

#-------------------------О проекте-------------------------
pipeline_expander = st.expander("Описание пайплайна:")
pipeline_expander.markdown(
    """
\nЗелёным обозначены этапы, корректировка которых доступна студенту, красным - этапы, которые предобработаны и скорректированы сотрудником лаборатории.
\n**1. Сбор данных:** был использован учебный набор данных по прогнозированию задолженности по выплатам заемщиков (**[ссылка](https://github.com/EgoVed/Reliability-of-the-borrower-)**)
\n**2. Предобработка данных:** состоит из 6 этапов
\n**2.1 Удаление столбцов:** очищаем данные от столбцов, которые не несут полезную информацию
\n**2.2 Обработка пропущенных значений:** заполняем пропуски в данных  
\n**2.3 Замена типов данных:** меняем типы данных
\n**2.4 Обработка текста:** приводим текстовые данные к общему виду, собираем их в группы 
\n**2.5 Обработка дубликатов:** проверяем данные на наличие дубликатов и избавляемся от них 
\n**2.6 Обработка артефактов:** очищаем данные от ошибок 
\n**2.7 Кодировка данных:** обрабатываем категориальные признаки двумя способами
\n**3.  Создание веб-приложения Streamlit:** оформление и выгрузка на сервер
\n**Используемые библиотеки:** [streamlit](https://docs.streamlit.io/library/get-started), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [sklearn](https://matplotlib.org/stable/api/index.html), 
[numpy](https://numpy.org/doc/stable/), [PIL](https://pillow.readthedocs.io/en/stable/), [pymystem3](https://github.com/nlpub/pymystem3)
""")

expand_bar = st.expander("Информация о датасете:")
expand_bar.markdown(
"""
\n**borr.csv** - исследование надежности заемщиков. Набор данных содержит личные сведения о каждом заемщике (возраст, высшее образование, информацию о семье, уровне дохода). 
Целевая переменная - была ли задолженность по возврату кредита (0 - задолженности не было; 1 - задолженость была)
""")

cols_info = st.expander('Описание столбцов:')
cols_info.markdown("""
\n**количество детей** - количество детей в семье
\n**трудовой стаж** - общий трудовой стаж в днях
\n**возраст** - возраст клиента в годах
\n**образование** - уровень образования клиента
\n**education_id** - идентификатор уровня образования
\n**семейный статус** - семейное положение
\n**family_status_id** - идентификатор семейного положения
\n**пол** - пол клиента
\n**тип занятости** - тип занятости
\n**уровень дохода** - ежемесячный доход
\n**цель кредита** - цель получения кредита
\n**просрочки** - имел ли задолженность по возврату кредитов
""")

my_data = pd.read_csv('borr.csv')

st.subheader('Блок 1: анализ данных')

st.write("""
Изучите данные, а затем ответьте на 10 вопросов 
""")

if st.checkbox('Показать Датасет'):
  st.dataframe(my_data)

if st.checkbox('Размер Датасета'):
  shape = st.radio(
    "Выбор данных",
     ('Строки', 'Столбцы'))
  if shape == 'Строки':
    st.write('Количество строк:', my_data.shape[0])
  elif shape == 'Столбцы':
    st.write('Количество столбцов:', my_data.shape[1])

if st.checkbox('Уникальные значения переменной'):
  cols = st.multiselect('Выбрать столбец', 
  my_data.columns.tolist())
  if cols:
    st.write(pd.DataFrame(my_data[cols].value_counts(), columns=['количество уникальных значений']))

if st.checkbox('Типы данных'):
  st.write('**Тип данных** - внутреннее представление, которое язык программирования использует для понимания того, как данные хранить и как ими оперировать')
  expander_bar = st.expander('Информация об основных типах данных')
  expander_bar.info('''Object - текстовые или смешанные числовые и нечисловые значения 
  \nINT - целые числа 
  \nFLOAT - дробные числа 
  \nBOOL - значения True/False
  \nDATETIME - значения даты и времени
  ''')

  st.write(pd.DataFrame(my_data.dtypes.astype('str'), columns=['тип данных']))

if st.checkbox('Описательная статистика по всем числовым столбцам'):
  describe_expander_ = st.expander('Информация о данных, которые входят в описательную статистику')
  describe_expander_.info('''Count - сколько всего было записей 
  \nMean - средняя велечина 
  \nStd - стандартное отклонение
  \nMin - минимальное значение
  \n25%/50%/70% - перцентили (показывают значение, ниже которого падает определенный процент наблюдений. Например, если число 5 - это 25% перцентиль, значит в наших данных 25% значений ниже 5)
  \nMax - максимальное значение
  ''')
  st.dataframe(my_data.describe())


non_val = st.checkbox('Пропущенные значения')
if non_val:
  st.write(pd.DataFrame(my_data.isnull().sum().sort_values(ascending=False), columns=['количество пропущенных значений']))


st.write("""
**Задание к 1 блоку:**
\n1. Какой номер у строки, в которой первым пропущено значение столбца "уровень дохода".
\n2. Сколько неповторяющихся значений в столбце "образование"? А сколько останется, если мы приведем все данные к единому типу написания?
\n3. На сколько категорий вы бы сгруппировали данные в столбце "цели кредита"? Перечислите названия групп (Здесь нет правильного ответа)
\n4. Среди заемщиков больше мужчин или женщин? В ответе укажите точное число.
\n5. Какой тип данных у столбца "уровень дохода"?
\n6. Только 25 процентов заемщиков зарабатывают больше 103 053 рублей. Это правда? Если нет, укажите точное число.
\n7. Какой минимальный возраст у заемщика?
\n8. Какое максимальное количество детей в семье? Сколько заемщиков с таким количеством?
\n9. Сколько в среднем работает заемщик? 
\n10. В каких столбцах есть пропущенные значения?
""")

#-----------------Preprocessing---------------

st.subheader('Блок 2: предобработка данных')
st.write('Мы внимательно посмотрели на данные и увидели там странные вещи. Попробуем теперь их обработать и привести к нормальному виду.')
st.warning('Обязательно выполняйте задания по порядку')

drop_col = st.checkbox('Шаг первый')
if drop_col:
  st.write("""
Для начала удалим столбец "Unnamed 0", так как он просто копирует индексы и не несет никакой информации.
Столбец 'трудовой стаж' тоже кажется не очень информативным. Мы знаем, что в нем хранятся данные о трудовом стаже в днях. Однако, увидели, что там есть отрицательные значения.
Также, если мы вспомним среднее количество рабочих дней и переведем это в количество лет, то у нас получится 175. Непонятно как обрабатывать эти данные. Удалим и этот столбец, чтобы он нам не мешал. """)
  col = st.multiselect('Столбцы',
  my_data.columns.tolist())
  drop = st.checkbox('Удалить')
  if drop:
    dropped_data = my_data.drop([col][0], axis=1) 
    st.dataframe(dropped_data)

drop_na = st.checkbox('Шаг второй')
if drop_na:
  st.write("""
  У нас осталось 2174 пропущенных значений в столбце "Уровень дохода". Есть разные способы обработки пропущенных значений: можно заменить данные нулями, 
  медианой (число, которое находится ровно посередине числового ряда),средним или модой (значение во множестве наблюдений, которое встречается наиболее часто).
  Кажется, что в нашем случае нули не очень подходят. Сперва посмотрим какие средние и медианные значения у нас принимает "уровень дохода" в зависимости от рода деятельности. 
  """)
  mean = dropped_data.groupby('тип занятости')['уровень дохода'].mean()
  st.write(pd.DataFrame(mean).set_axis(['Среднее значение уровня дохода'], axis='columns'))
  median = dropped_data.groupby('тип занятости')['уровень дохода'].median()
  st.write(pd.DataFrame(median).set_axis(['Медианное значение уровня дохода'], axis='columns'))
  st.write("""
  Оба значения примерно совпадают, поэтому можно выбрать любой из двух.
  """)
  fill_none = st.selectbox("На что меняем?", 
                          ("Среднее", "Медиана"))
  if fill_none == 'Среднее':
    mean_ = dropped_data.groupby('тип занятости')['уровень дохода'].transform('mean')
    dropped_data['уровень дохода'].fillna(mean_, inplace=True)
  if fill_none == 'Медиана':  
    med = dropped_data.groupby('тип занятости')['уровень дохода'].transform('median')
    dropped_data['уровень дохода'].fillna(med, inplace=True)
  show_na = st.checkbox('Показать пропущенные значения:')
  if show_na:
    st.write('Теперь мы видим, что у нас больше нет пропущенных значений в датасете.')
    st.write(pd.DataFrame(dropped_data.isnull().sum().sort_values(ascending=False), columns=['количество пропущенных значений']))
  

change_type = st.checkbox('Шаг третий')
if change_type:
  st.write("""
  Мы уже отмечали, что у столбца "уровень дохода" тип float. Кажется, было бы удобней смотреть на зарплату в нормальном виде.
  Давайте заменим тип данных столбца "уровень дохода" на int64.
  """)
  col = st.selectbox('Столбец', dropped_data.columns.tolist())
  typ = st.selectbox('Тип', dropped_data.dtypes.unique())
  dropped_data[col] = dropped_data[col].astype(typ)
  if st.checkbox('показать'):
    st.write('Вот так теперь выглядят данные в столбце "уровень дохода"')
    st.dataframe(dropped_data['уровень дохода'])

similar = st.checkbox('Шаг четвертый')
if similar:
  st.write("""
  Мы с вами видели, что в столбце "образование" есть повторяющиеся значения, которые написаны по разному.
  В столбце "цель кредита" похожая ситуация. Давайте приведем наши данные к общему виду: в столбце "образование" приведем все значения к нижнему регистру, 
  а в столбце "цель кредита" заменим значения на 4 группы (жилье, автомобиль, обучение и личные цели)
  """)
  if st.checkbox('Привести к нижнему регистру'):
    dropped_data['образование'] = dropped_data['образование'].str.lower()
    st.write(dropped_data['образование'].value_counts())
  if st.checkbox('Разбить на 4 группы'):
    purpose_lem = dropped_data['цель кредита'].apply(lemmatization)
    dropped_data['цель кредита'] = purpose_lem
    st.write(dropped_data['цель кредита'].value_counts())

rm_duplicates = st.checkbox('Шаг пятый')
if rm_duplicates:
  st.write("Теперь проверим датасет на наличие дубликатов (полностью похожих друг на друга строк)")
  if st.checkbox('Сколько всего дубликатов'):
    dupl = dropped_data.duplicated().sum()
    st.write(f'Всего {dupl} дубликат')
  st.write('Удалим повторящиеся строки')
  if st.checkbox('Сколько осталось дубликатов'):
    dropped_data = dropped_data.drop_duplicates().reset_index(drop=True)
    new_dupl = dropped_data.duplicated().sum()
    st.write(f'Всего {new_dupl} дубликат')
  
artefacts = st.checkbox('Шаг шестой')
if artefacts:
  st.write("""Обработаем ошибки в данных. Изучив датасет, мы заметили странные значения в столбцах "пол", "количество детей" и "возраст". 
  Давайте удалим строку, где в столбце пол указан "XNA" и строку, где в столбце "возраст" стоит 0, а в столбце "колчество детей" заменим значения -1 и 20  на 1 и 2.""")
  if st.checkbox('Удалить артефакт из столбца "пол"'):
    dropped_data=dropped_data.drop(dropped_data[dropped_data['пол']=='XNA'].index)
    st.write(dropped_data['пол'].value_counts())

  if st.checkbox('Удалить артефакт из столбца "возраст"'):
    dropped_data=dropped_data.drop(dropped_data[dropped_data['возраст']==0].index)
    st.write(dropped_data['возраст'].value_counts())

  if st.checkbox('Заменить артефакты в столбце "количество детей"'):
    dropped_data['количество детей']=dropped_data['количество детей'].replace(-1,1)
    dropped_data['количество детей']=dropped_data['количество детей'].replace(20,2)
    st.write(dropped_data['количество детей'].value_counts())


categorical_features = st.checkbox('Шаг седьмой')
if categorical_features:
  st.write("""
  Попробуем несколько способов обработки категориальных признаков. Эта процедура называется кодировкой. Она нужна, чтобы привести категориальные признаки к численному виду. 
  Во-первых, посмотрите на столбец "пол". Это бинарный признак, поэтому мы можем заменить M, F на 0 и 1
  """)
  categorical = st.expander('Что такое категориальные признаки?')
  categorical.markdown('Категориальные признаки - те, которые не имеют численного представления. Могут иметь как 2 уникальных значения (бинарные признаки), так и более')
  change_binary = st.checkbox('Обработать столбец "пол"')
  if change_binary:
    dropped_data.пол = dropped_data.пол.map(dict(F=1, M=0))
    st.write(dropped_data)
    st.write("""
    Что делать, если у нас больше, чем 2 уникальных значений признака? Можно попробовать применить **[быстрое кодирование (One hot encoding)](https://www.helenkapatsa.ru/bystroie-kodirovaniie/)**
    С помощью one-hot мы конвертируем каждое категориальное значение в новый категориальный столбец и присваиваем этим столбцам двоичное значение 1 или 0. Давайте посмотрим как это выглядит на примере
    кодировки столбца "образование" (новые столбцы появятся в конце).
    \n2 статьи про разные способы кодировки данных: **[1](https://dyakonov.org/2016/08/03/python-категориальные-признаки/)**, **[2](https://habr.com/ru/post/666234/)**
    """)
    one_hot_en = st.checkbox('Применить горячее кодирование к столбцу "образование"')
    if one_hot_en:
      dropped_data = pd.get_dummies(dropped_data, columns=['образование'])
      st.write(" Обратите внимание, что сам столбец 'образование' исчез из нашего датасета, вместо него появились 5 новых в конце датасета")
      st.write(dropped_data)

expander = st.expander("Заключение:")
expander.markdown("""
В первой части лабораторной работы мы познакомились с некоторыми возможностями предобработки данных. Во второй части мы проверим, как предобработка данных повляет на результат обучения модели.
""")



