import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
RU_STOP = set(stopwords.words('russian'))
EN_STOP = set(stopwords.words('english'))
CUSTOM_STOP = {"это", "для", "на", "в", "про", "кино", "фильм"}
ALL_STOP = RU_STOP | EN_STOP | CUSTOM_STOP

def generate_all_charts(db_path):
    plt.style.use('ggplot')
    engine = create_engine(f'sqlite:///{db_path}')

    def format_xticks(ax):
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(45)
            lbl.set_ha('right')

    # ─── 1) Склеиваем события и подтягиваем lang для каждого user_id ───
    df_fb = pd.read_sql("SELECT date(timestamp) AS day, user_id, lang FROM feedback", engine)
    df_fb['day'] = pd.to_datetime(df_fb['day'])
    user_lang = df_fb.groupby('user_id')['lang'].first().rename('lang').reset_index()

    df_act = pd.read_sql("SELECT date(ts) AS day, user_id FROM actions", engine)
    df_act['day'] = pd.to_datetime(df_act['day'])

    df_days = pd.concat([df_fb[['day','user_id']], df_act[['day','user_id']]], ignore_index=True)
    df_days = df_days.merge(user_lang, on='user_id', how='inner')

    # ─── 2) DAU/WAU/MAU по каждому lang ───
    for period, name, xtick_freq in [
        ('D', 'dau', 'D'),
        ('W-MON', 'wau', 'W-MON'),
        ('M', 'mau', 'M'),
    ]:
        fig, ax = plt.subplots(figsize=(10,4))
        for lang in ['ru','en','kk']:
            df_l = df_days[df_days['lang'] == lang].copy()
            if period == 'D':
                grp = df_l.groupby('day')['user_id'].nunique()
                x = grp.index
                y = grp.values
            else:
                df_l['period'] = df_l['day'].dt.to_period(period)
                grp = df_l.groupby('period')['user_id'].nunique()
                x = grp.index.to_timestamp()  # начало периода
                y = grp.values

            ax.plot(x, y, marker='o', linestyle='-', label=lang.upper())

        ax.set_title(f'{name.upper()} по языкам')
        ax.set_xlabel('Дата')
        ax.set_ylabel(name.upper())
        format_xticks(ax)
        ax.legend(title='lang')
        fig.tight_layout()
        fig.savefig(f'chart_{name}_by_lang.png')
        plt.close(fig)

    # ─── 3) Retention curve ───
    first_day = df_days.groupby('user_id')['day'].min().rename('first_day')
    df_ret = df_days.join(first_day, on='user_id')
    df_ret['delta_days'] = (df_ret['day'] - df_ret['first_day']).dt.days
    first_day = df_days.groupby('user_id')['day'].min()

    total_users = len(first_day)

    ret = df_ret[df_ret['delta_days'] <= 30] \
        .drop_duplicates(['user_id', 'delta_days']) \
        .groupby('delta_days')['user_id'] \
        .nunique() \
        .rename('n_active') \
        .to_frame()
    ret['total'] = total_users
    ret['retention'] = ret['n_active'] / ret['total']

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ret.index, ret['retention'], marker='o', linestyle='-')
    ax.set_title('Retention Curve (доля вернувшихся пользователей)')
    ax.set_xlabel('Дней с момента первого визита')
    ax.set_ylabel('Retention')
    fig.tight_layout()
    fig.savefig('chart_retention_curve.png')
    plt.close(fig)

    # ─── 4) Lifetime Value — длина жизни пользователя в днях ───
    user_span = df_days.groupby('user_id')['day'] \
                       .agg(['min','max']).rename(columns={'min':'first','max':'last'})
    user_span['lifetime_days'] = (user_span['last'] - user_span['first']).dt.days + 1

    fig, ax = plt.subplots(figsize=(10,4))
    ax.hist(user_span['lifetime_days'], bins=30)
    avg_lt = user_span['lifetime_days'].mean()
    ax.set_title(f'Распределение LTV (средняя жизнь: {avg_lt:.1f} дней)')
    ax.set_xlabel('Lifetime (дней)')
    ax.set_ylabel('Число пользователей')
    fig.tight_layout()
    fig.savefig('chart_lifetime_value.png')
    plt.close(fig)

    # ─── 5) Топ-20 слов в комментариях с NLTK-стоп-словами ───
    df_com = pd.read_sql("SELECT comment FROM feedback WHERE comment IS NOT NULL", engine)
    text = " ".join(df_com['comment'].astype(str)).lower()
    words = re.findall(r"\b[а-яёa-z]{3,}\b", text)
    filtered = [w for w in words if w not in ALL_STOP]
    top20 = Counter(filtered).most_common(20)
    words_, counts_ = zip(*top20) if top20 else ([], [])

    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(words_, counts_)
    ax.set_title('Топ-20 слов в комментариях (без стоп-слов)')
    ax.set_ylabel('Частота')
    format_xticks(ax)
    fig.tight_layout()
    fig.savefig('chart_top_words_filtered.png')
    plt.close(fig)

    print("Сохранены графики:\n"
          " chart_dau_by_lang.png, chart_wau_by_lang.png, chart_mau_by_lang.png,\n"
          " chart_retention_curve.png, chart_lifetime_value.png,\n"
          " chart_top_words_filtered.png")

if __name__ == '__main__':
    generate_all_charts('C:/Users/user/PyCharmProjects/pythonProject79/users.db')