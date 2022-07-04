from nbformat import write
import streamlit as st
import pandas as pd #Пандас
import matplotlib.pyplot as plt #Отрисовка графиков
import seaborn as sns
import numpy as np #Numpy
from PIL import Image
import time
from datetime import datetime 
from pymystem3 import Mystem
from collections import Counter

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
            >Предварительная обработка данных</h1>''', 
            unsafe_allow_html=True)

image = Image.open('images/Pipeline_1.png')
st.image(image)

st.write("""
Данный стримлит предназначен для демонстрации способов предварительной обработки данных (Data Preprocessing).
""")
#-------------------------О проекте-------------------------
expander_bar = st.expander("Перед тем, как начать:")
expander_bar.markdown(
    """
\nПредварительная обработка данных - важный этап в науке о данных. Для того, чтобы обучать модель, ей необходимо предоставить очищенные данные
в понятном ей виде. Предварительная обработка данных, на равне с разведочным анализом данных, обычно занимает большую часть проекта. В этом стримлите мы попробуем очистить данные, сгруппировать и
привести к нужному типу. 
\n**Используемые библиотеки:** [streamlit](https://docs.streamlit.io/library/get-started), [pandas](https://pandas.pydata.org/docs/user_guide/index.html), [matplotlib](https://matplotlib.org/stable/api/index.html), [seaborn](https://seaborn.pydata.org).
\n **Полезно почитать:** [1](https://ru.wikipedia.org/wiki/Предварительная_обработка_данных), [2](https://habr.com/ru/post/511132/), [3](https://pythobyte.com/data-preprocessing-0cb9135c/)
""")

expand_bar = st.expander("Информация о датасете:")
expand_bar.markdown(
"""
\n**borr.csv** - исследование надежности заемщиков. Набор данных содержит личные сведения о каждом заемщике (возраст, высшее образование, информацию о семье, уровне дохода). 
Целевая переменная - была ли задолженность по возврату кредита (0 - задолженности не было; 1 - задолженость была)
""")
my_data = pd.read_csv('borr.csv')
 
st.subheader('Посмотрим на данные')

if st.checkbox('Показать Датасет'):
  #number = st.number_input('Сколько строк показать', min_value=1, max_value=my_data.shape[1])
  #st.dataframe(my_data.head(number))
  st.dataframe(my_data)

if st.checkbox('Размер Датасета'):
  shape = st.radio(
    "Выбор данных",
     ('Строки', 'Колонки'))
  if shape == 'Строки':
    st.write('Количество строк:', my_data.shape[0])
  elif shape == 'Колонки':
    st.write('Количество колонок:', my_data.shape[1])

if st.checkbox('Уникальные значения переменной'):
  cols = st.multiselect('Выбрать колонку', 
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

# if st.checkbox('Описательная статистика по всем числовым колонкам'):
#   expander_bar = st.expander('Информация о данных, которые входят в описательную статистику')
#   expander_bar.info('''Count - сколько всего было записей 
#   \nMean - средняя велечина 
#   \nStd - стандартное отклонение
#   \nMin - минимальное значение
#   \n25%/50%/70% - перцентили (показывают значение, ниже которого падает определенный процент наблюдений. Например, если число 5 - это 25% перцентиль, значит в наших данных 25% значений ниже 5)
#   \nMax - максимальное значение
#   ''')
#   st.dataframe(my_data.describe())


non_val = st.checkbox('Пропущенные значения')
if non_val:
  st.write(pd.DataFrame(my_data.isnull().sum().sort_values(ascending=False), columns=['количество пропущенных значений']))

#-----------------Preprocessing---------------

st.subheader('Предобработаем данные')
st.warning('Обязательно выполняйте задания по порядку')

drop_col = st.checkbox('Удалить столбец')
if drop_col:
  st.write("""
Для начала удалим столбец "Unnamed 0", так как он просто копирует индексы и не несет никакой информации.
Столбец 'трудовой стаж' тоже кажется не очень информативным. Мы не знаем, откуда там отрицательные значения и как их трактовать. 
Удалим и этот столбец, чтобы он нам не мешал. """)
  col = st.multiselect('Колонки',
  my_data.columns.tolist())
  drop = st.checkbox('Удалить')
  if drop:
    dropped_data = my_data.drop([col][0], axis=1) 
    st.dataframe(dropped_data)

drop_na = st.checkbox('Обработаем пропущенные значения')
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
  Оба значения примерно совпадают, но медиана дает более точную информацию. Заменим наши пропущенные значения на медианные 
  """)
  median_change = st.checkbox('Заменить на медианное значение')
  # dropped_data['уровень дохода'].fillna(median, inplace=True)
  if median_change:
    med = dropped_data.groupby('тип занятости')['уровень дохода'].transform('median')
    dropped_data['уровень дохода'].fillna(med, inplace=True)
    #st.dataframe(dropped_data)
    st.write('Теперь мы видим, что у нас больше нет пропущенных значений в датасете.')
    st.write(pd.DataFrame(dropped_data.isnull().sum().sort_values(ascending=False), columns=['количество пропущенных значений']))
  

change_type = st.checkbox('Изменить тип данных')
if change_type:
  st.write("""
  Если взглянуть сверху на типы данных, можно увидеть, что у столбца "уровень дохода" тип float. Кажется, было бы удобней смотреть на зарплату в нормальном виде.
  Давайте заменим тип данных столбца "уровень дохода" на int64.
  """)
  col = st.selectbox('Колонка', dropped_data.columns.tolist())
  typ = st.selectbox('Тип', dropped_data.dtypes.unique())
  dropped_data[col] = dropped_data[col].astype(typ)
  if st.checkbox('показать'):
    st.write('Вот так теперь выглядят данные в столбце "уровень дохода"')
    st.dataframe(dropped_data['уровень дохода'])

similar = st.checkbox('Обработка похожих значений')
if similar:
  # dupl = dropped_data.duplicated().sum()
  # st.write(dupl)
  st.write("""
  Внимательно посмотрите на уникальные значения в столбце "образование". Там есть повторяющиеся значения, но они записаны по разному.
  Тоже самое можно заметить в столбце "цель кредита". Давайте приведем наши данные к общему виду: в столбце "образование" приведем все строки к нижнему регистру, 
  а в столбце "цель кредита" заменим данные на 4 группы (жилье, автомобиль, обучение и личные цели)
  """)
  if st.checkbox('Привести к нижнему регистру'):
    dropped_data['образование'] = dropped_data['образование'].str.lower()
    st.write(dropped_data['образование'].value_counts())
  if st.checkbox('Разбить на 4 группы'):
    purpose_lem = dropped_data['цель кредита'].apply(lemmatization)
    dropped_data['цель кредита'] = purpose_lem
    st.write(dropped_data['цель кредита'].value_counts())

  # st.write("Теперь проверим датасет на наличие дубликатов (полностью похожих друг на друга строк)")
  # if st.checkbox('Сколько всего дубликатов'):
  #   dupl = dropped_data.duplicated().sum()
  #   st.write(f'Всего {dupl} дубликат')
  # st.write('Удалим повторящиеся колонки')
  # if st.checkbox('Сколько осталось дубликатов'):
  #   dropped_data = dropped_data.drop_duplicates().reset_index(drop=True)
  #   new_dupl = dropped_data.duplicated().sum()
  #   st.write(f'Всего {new_dupl} дубликат')

rm_duplicates = st.checkbox('Обработка дубликатов')
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
  
artefacts = st.checkbox('Избавимся от ошибок в данных')
if artefacts:
  st.write("""Посмотрите на уникальные значения столбцов "пол", "количество детей", "возраст". Мы видим там странные значение. 
  Давайте удалим строку, где в столбце пол указан "XNA", строку, где в столбце "возраст" стоит 0 
  и в столбце "колчество детей" заменим значения -1 и 20  на 1 и 2.""")
  if st.checkbox('Удалить артефакт из столбца "пол"'):
    dropped_data=dropped_data.drop(dropped_data[dropped_data['пол']=='XNA'].index)
    st.write(dropped_data['пол'].value_counts())
  if st.checkbox('Удалить артефакт из столбца "возраст"'):
    dropped_data=dropped_data.drop(dropped_data[dropped_data['возраст']==0].index)
    st.write(dropped_data['возраст'].value_counts())
  if st.checkbox('Заменить артефакты в столбце "количество детей"'):
    dropped_data['количество детей']=dropped_data['количество детей'].replace(-1,1) # заменяем знечение в столбце ['children'] -1 на 1 и 20 на 2. 
    dropped_data['количество детей']=dropped_data['количество детей'].replace(20,2) # скорее всего это были неправильно занесенные данные
    st.write(dropped_data['количество детей'].value_counts())

final = st.checkbox('Посмотрим на измененный датасет')
if final:
  st.write("""
Мы сделали довольно много преобразований, почистили наши данные и избавились от ненужной информации. Самое время посмотреть на итоговую таблицу
""")
  #st.write(my_data.info())
  st.write(dropped_data)

expander = st.expander("Заключение:")
expander.markdown("""
Мы проделали большую работу с нашим исходным датасетом, но этимии возможностями предобработка данных не исчерпывается. Обработка категориальных (все нечисловые значения, например слова) переменных 
([самый простой способ](https://www.helenkapatsa.ru/bystroie-kodirovaniie/), [статья про разные виды кодировок](https://dyakonov.org/2016/08/03/python-категориальные-признаки/)) 
""")



