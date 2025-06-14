import json
import logging
import re
import sqlite3
import requests
import asyncio
import difflib
from telegram import Bot, Update, InlineKeyboardButton as Btn, InlineKeyboardMarkup as Mk
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler,
    CallbackQueryHandler, MessageHandler, filters, ConversationHandler
)
from telegram.error import BadRequest
from transformers import pipeline
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
import pickle, faiss
from transformers import MarianMTModel, MarianTokenizer
from functools import partial
from contextlib import contextmanager
# Пути к файлам
INDEX_PATH = "movies.index"
META_PATH  = "movies_meta.pkl"
import config
# ───────────────────────── Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

# ───────────────────────── Clear webhook before polling
async def clear_webhook():
    await Bot(config.TELEGRAM_TOKEN).delete_webhook(drop_pending_updates=True)
asyncio.get_event_loop().run_until_complete(clear_webhook())
log.info("Webhook cleared — ready for polling")

# ───────────────────────── Load messages
with open("messages.json", encoding="utf-8") as f:
    M = json.load(f)
GENRES = {
    "ru": [("Боевик",28), ("Комедия",35), ("Драма",18),
           ("Ужасы",27), ("Мелодрама",10749), ("Фантастика",878)],
    "en": [("Action",28), ("Comedy",35), ("Drama",18),
           ("Horror",27), ("Romance",10749), ("Sci-Fi",878)],
    "kk": [("Қылмыстық",28), ("Комедия",35), ("Драма",18),
           ("Қорқынышты",27), ("Мелодрама",10749), ("Ғылыми фантастика",878)],
}


# ───────────────────────── NLP pipelines (synchronous)
model_emb = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
tok_mt = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
model_mt = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
mask_pipe = pipeline(
    "fill-mask",
    model="xlm-roberta-base",
    tokenizer="xlm-roberta-base"
)
pipe_en = pipe_ru = pipe_kz = mask_pipe
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)


def query_emb(text: str):
    vec = model_emb.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    return vec

def super_search(query: str, k: int = 10):
    en = translate_to_en(query)
    qv = query_emb(en)
    D, I = index.search(qv, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        m = meta[idx]
        hybrid = 0.7 * float(score) + 0.3 * (m.get('vote_average', 0) / 10)
        results.append({**m, "score": float(score), "hybrid": hybrid})
    return sorted(results, key=lambda x: x["hybrid"], reverse=True)

async def super_search_async(text: str, k: int = 10):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(super_search, text, k))

def _semantic_hint_sync(text: str, lang: str) -> str:
    pipe = {"kk": pipe_kz, "ru": pipe_ru, "en": pipe_en}[lang]
    try:
        seq = pipe(f"{text} <mask>")[0]["sequence"]
        return seq.replace("<mask>", "").strip()
    except Exception:
        return text

async def semantic_hint(text: str, lang: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _semantic_hint_sync, text, lang)

def translate_to_en(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

# ───────────────────────── Feedback DB
con = sqlite3.connect(
    "users.db",
    check_same_thread=False,
    timeout=30,
    isolation_level=None,
)
con.execute("PRAGMA busy_timeout = 30000;")
con.execute("PRAGMA journal_mode = WAL;")
con.execute("PRAGMA synchronous = NORMAL;")

cur = con.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS feedback(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER, lang TEXT,
    rating TEXT, comment TEXT, age INTEGER,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS actions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER,
  action TEXT,
  details TEXT,
  ts DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
con.commit()


# ───────────────────────── Logging helper & decorator
def log_action(user_id: int, action: str, details: str = ""):
    cur.execute(
        "INSERT INTO actions(user_id, action, details) VALUES (?, ?, ?)",
        (user_id, action, details)
    )
    con.commit()

def with_logging(action_name):
    def decorator(func):
        async def wrapper(update, ctx, *args, **kwargs):
            user_id = update.effective_user.id
            details = (update.callback_query.data
                       if update.callback_query
                       else getattr(update.message, "text", ""))
            log_action(user_id, action_name, details)
            return await func(update, ctx, *args, **kwargs)
        return wrapper
    return decorator

# ───────────────────────── TMDB helpers
TMDB_API       = "https://api.themoviedb.org/3"
TMDB_MOVIE_URL = "https://www.themoviedb.org/movie/"
LANG_CODES     = {"ru":"ru-RU","en":"en-US","kk":"kk-KZ"}

async def tmdb_get(endpoint: str, **params) -> dict:
    params["api_key"] = config.TMDB_API_KEY
    url = TMDB_API + endpoint
    loop = asyncio.get_running_loop()
    def fetch():
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.error("TMDB error: %s", e)
            return {}
    return await loop.run_in_executor(None, fetch)

async def discover(**params):
    data = await tmdb_get("/discover/movie", **params)
    return data.get("results", [])

async def movie_details(mid: int):
    return await tmdb_get(f"/movie/{mid}")

# ───────────────────────── Semantic-search model
sem_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

async def semantic_search_fallback(text: str, lang: str) -> list:
    movies = []
    for p in range(1, 4):
        movies += await discover(sort_by="popularity.desc", page=p)
    overviews = [m.get("overview","") or "" for m in movies]
    loop = asyncio.get_running_loop()
    embeds = await loop.run_in_executor(None, lambda: sem_model.encode(overviews, convert_to_tensor=True))
    q_emb = await loop.run_in_executor(None, lambda: sem_model.encode(translate_to_en(text).lower(), convert_to_tensor=True))
    sims = util.cos_sim(q_emb, embeds)[0]
    topk = sims.topk(k=10).indices.cpu().numpy()
    return [movies[i] for i in topk]

async def search_by_reviews(text: str):
    en_text = translate_to_en(text).lower()
    matches = []
    for page_num in range(1, 4):
        page = await discover(sort_by="popularity.desc", page=page_num)
        for m in page:
            revs = await tmdb_get(f"/movie/{m['id']}/reviews")
            for r in revs.get("results", []):
                if en_text in r.get("content", "").lower():
                    matches.append(m)
                    break
    seen, uniq = set(), []
    for m in matches:
        if m["id"] not in seen:
            seen.add(m["id"])
            uniq.append(m)
    return uniq

async def search_by_description(text: str, lang: str):
    code = LANG_CODES[lang]
    en = translate_to_en(text).strip().lower()

    # 0) by reviews
    rv = await search_by_reviews(text)
    if rv:
        return rv

    # 1) person search
    prs = (await tmdb_get("/search/person", query=en, language=code)).get("results", [])
    if prs:
        m = await discover(with_cast=prs[0]["id"], sort_by="popularity.desc")
        if m: return m

    # 2) hint → person
    hint = await semantic_hint(en, "en")
    prs2 = (await tmdb_get("/search/person", query=hint, language=code)).get("results", [])
    if prs2:
        m = await discover(with_cast=prs2[0]["id"], sort_by="popularity.desc")
        if m: return m

    # 3) multi-search
    multi = (await tmdb_get("/search/multi", query=hint, language=code)).get("results", [])
    mov = [x for x in multi if x.get("media_type") == "movie"]
    if mov: return mov

    # 4) movie-search hint
    m4 = (await tmdb_get("/search/movie", query=hint, language=code)).get("results", [])
    if m4: return m4

    # 5) movie-search en
    m5 = (await tmdb_get("/search/movie", query=en, language=code)).get("results", [])
    if m5: return m5

    # 6) split parts
    seen, res = set(), []
    for part in re.split(r"[.,;]", en):
        p = part.strip()
        if len(p) < 4: continue
        for x in (await tmdb_get("/search/movie", query=p, language=code)).get("results", []):
            if x["id"] not in seen:
                seen.add(x["id"])
                res.append(x)
    if res:
        res.sort(key=lambda x: x.get("popularity", 0), reverse=True)
        return res

    # 7) keywords
    STOPWORDS = set("the a an and or of in on at to with for from by это для на в фильм кино про бұл осы".split())
    toks = re.findall(r"[A-Za-zА-Яа-яҰұҚқҒғӨөҺһІіЁё]{4,}", text.lower())
    kw = [t for t in toks if t not in STOPWORDS][:6]
    if kw:
        kw_ids = []
        for t in kw:
            r = (await tmdb_get("/search/keyword", query=t)).get("results", [])
            if r: kw_ids.append(str(r[0]["id"]))
        if kw_ids:
            kres = await discover(with_keywords=",".join(kw_ids), sort_by="popularity.desc")
            if kres: return kres

    # 8) overview fallback
    top = await discover(sort_by="popularity.desc", page=1)
    out = [m for m in top if en in (m.get("overview","") or "").lower()]
    if out: return out

    # 9) semantic fallback
    sem = await semantic_search_fallback(text, lang)
    if sem:
        return sem

    return []

# ───────────────────────── States & UI
(LANG, MENU, GENRE, ACTOR_MAIN, ACTOR_CUSTOM, YEAR, RATING, COMMENT, AGE, CUSTOM_GENRE, CUSTOM_YEAR, CUSTOM_DESC) = range(12)
LABELS = {
    "ru": {"rating":"Рейтинг","runtime":"Длительность","like":"Понравилось","dislike":"Не понравилось","cmt":"Оставить комментарий","thanks":"Спасибо!"},
    "en": {"rating":"Rating","runtime":"Runtime","like":"Liked it","dislike":"Didn't like","cmt":"Leave a comment","thanks":"Thanks!"},
    "kk": {"rating":"Рейтинг","runtime":"Ұзақтығы","like":"Ұнады","dislike":"Ұнамады","cmt":"Пікір қалдыру","thanks":"Рақмет!"}
}

def action_kb(lang: str):
    return Mk([
        [Btn(M["buttons"]["repeat"][lang], callback_data="repeat_last")],
        [Btn(M["buttons"]["home"][lang],   callback_data="go_menu")],
    ])

def finish_kb(lang: str):
    return Mk([
        [Btn(M["buttons"]["finish"][lang], callback_data="finish")],
        [Btn(M["buttons"]["repeat"][lang], callback_data="repeat_last")],
        [Btn(M["buttons"]["home"][lang],   callback_data="go_menu")],
    ])

async def send_movie(update: Update, movie: dict, lang: str):
    det     = await movie_details(movie["id"]) or {}
    runtime = det.get("runtime","—")
    year    = movie.get("release_date","")[:4]
    L       = LABELS[lang]
    txt = (
        f"<b>{movie['title']}</b> ({year})\n"
        f"{L['rating']}: {movie['vote_average']}\n"
        f"{L['runtime']}: {runtime} мин.\n\n"
        f"{movie.get('overview','')}\n"
        f"{TMDB_MOVIE_URL}{movie['id']}"
    )
    await update.effective_chat.send_message(txt, parse_mode="HTML", reply_markup=finish_kb(lang))

async def send_top10(update: Update, movies: list, lang: str):
    txt = "\n".join(
        f"{i+1}. [{m['title']}]({TMDB_MOVIE_URL}{m['id']}) ({m.get('release_date','')[:4]}) – ⭐ {m['vote_average']}"
        for i,m in enumerate(movies)
    )
    await update.effective_chat.send_message(txt, parse_mode="Markdown", reply_markup=action_kb(lang))

async def ask_rating(update: Update, lang: str):
    L = LABELS[lang]
    kb = [
        [Btn(L["like"],    callback_data="rate_like"), Btn(L["dislike"], callback_data="rate_dislike")],
        [Btn(L["cmt"],     callback_data="rate_comment")]
    ]
    await update.effective_chat.send_message(M["rate_bot"][lang], reply_markup=Mk(kb))
    return RATING

async def finish_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    lang = ctx.user_data["lang"]
    return await ask_rating(update, lang)

async def show_menu(obj, lang: str):
    kb = [
        [Btn(M["buttons"]["genre"][lang], callback_data="by_genre"),
         Btn(M["buttons"]["actor"][lang], callback_data="by_actor")],
        [Btn(M["buttons"]["year"][lang], callback_data="by_year"),
         Btn(M["buttons"]["top10"][lang], callback_data="top10")],
        [Btn(M["buttons"]["custom_search"][lang], callback_data="custom_search")],
        [Btn(M["buttons"]["change_lang"][lang], callback_data="change_lang")],
        [Btn(M["buttons"]["contact_us"][lang], callback_data="contact_us")],
        [Btn(M["buttons"]["our_site"][lang],
             url="https://moviesrecommendation-wsetxewvgnzksr6sanzarh.streamlit.app/")]
    ]
    text = M["main_menu"][lang]
    if hasattr(obj, "edit_message_text"):
        await obj.edit_message_text(text, reply_markup=Mk(kb))
    else:
        await obj.reply_text(text, reply_markup=Mk(kb))


# ───────────────────────── Handlers

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    kb = [[
        Btn("🇰🇿", callback_data="lang_kk"),
        Btn("🇷🇺", callback_data="lang_ru"),
        Btn("🇺🇸", callback_data="lang_en"),
    ]]
    await update.message.reply_text(
        f"{M['greeting']['en']}\n{M['choose_language']['en']}",
        reply_markup=Mk(kb)
    )
    return LANG

async def set_lang(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try: await update.callback_query.answer()
    except BadRequest: pass
    lang = update.callback_query.data.split("_")[1]
    ctx.user_data["lang"] = lang
    await show_menu(update.callback_query, lang)
    return MENU

async def menu_router(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try: await update.callback_query.answer()
    except BadRequest: pass
    lang, q = ctx.user_data["lang"], update.callback_query.data
    if   q=="by_genre":      return await genre_menu(update, ctx)
    elif q == "by_actor":
        return await actor_menu(update, ctx)
    elif q=="by_year":       return await year_menu(update, ctx)
    elif q=="top10":         return await top10(update, ctx)
    elif q == "custom_search":
        return await start_custom(update, ctx)
    elif q=="change_lang":
        await update.callback_query.edit_message_text(
            f"{M['greeting']['en']}\n{M['choose_language']['en']}",
            reply_markup=Mk([[Btn("🇰🇿", callback_data="lang_kk"),
                              Btn("🇷🇺", callback_data="lang_ru"),
                              Btn("🇺🇸", callback_data="lang_en")]]))
        return LANG
    elif q == "contact_us":
        await update.callback_query.edit_message_text(
            "@blinkotenok\n@Lia_Yng\n@T_Dikko",
            reply_markup=Mk([[Btn(M["buttons"]["back"][ctx.user_data['lang']], callback_data="go_menu")]]))
        return MENU
    elif q=="go_menu":
        await show_menu(update.callback_query, lang)
        return MENU
    return MENU


# Genre
async def genre_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lang = ctx.user_data["lang"]
    kb = [[Btn(name, callback_data=f"genre_{gid}")] for name, gid in GENRES[lang]]
    kb.append([Btn(M["buttons"]["back"][lang], callback_data="go_menu")])
    await update.callback_query.edit_message_text(
        M["select_genre"][lang], reply_markup=Mk(kb)
    )
    return GENRE

@with_logging("pick_genre")
async def genre_pick(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try: await update.callback_query.answer()
    except BadRequest: pass
    lang = ctx.user_data["lang"]
    gid  = int(update.callback_query.data.split("_")[1])
    await update.callback_query.message.chat.send_action(ChatAction.TYPING)
    movies = await discover(
        with_genres=gid,
        sort_by="popularity.desc",
        language=LANG_CODES[lang]
    )
    if not movies:
        await update.callback_query.edit_message_text(M["no_results"][lang])
        return MENU
    ctx.user_data["last_query"] = {"movies": movies, "type": "genre", "index": 0}
    await send_movie(update, movies[0], lang)
    return GENRE

# Actor
@with_logging("actor_menu")
async def actor_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lang = ctx.user_data["lang"]
    await update.callback_query.edit_message_text(
        "Введите имя актёра для обычного поиска:",
        reply_markup=Mk([[Btn("⬅️ Назад", callback_data="go_menu")]])
    )
    return ACTOR_MAIN

async def actor_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try: await update.callback_query.answer()
    except BadRequest: pass
    lang = ctx.user_data["lang"]
    await update.callback_query.message.chat.send_action(ChatAction.TYPING)
    movies = await discover(sort_by="popularity.desc")
    ctx.user_data["last_query"] = {"movies": movies, "type": "actor_none", "index": 0}
    await send_movie(update, movies[0], lang)
    return ACTOR

async def actor_main_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lang = ctx.user_data["lang"]
    hint = update.message.text.strip()
    await update.message.chat.send_action(ChatAction.TYPING)
    raw_hint = update.message.text.strip()
    hint_en  = translate_to_en(raw_hint)
    prs_data = await tmdb_get("/search/person", query=raw_hint, language="en-US")
    prs = prs_data.get("results", [])
    if not prs:
        prs_data = await tmdb_get("/search/person", query=hint_en, language="en-US")
        prs = prs_data.get("results", [])
    if not prs:
        movies = await discover(sort_by="popularity.desc")
    else:
        exact = [p for p in prs if p["name"].lower() in (raw_hint.lower(), hint_en.lower())]
        if exact:
            match = exact[0]
        else:
            names = [p["name"] for p in prs]
            close = (difflib.get_close_matches(raw_hint, names, n=1, cutoff=0.6)
                     or difflib.get_close_matches(hint_en, names, n=1, cutoff=0.6))
            if close:
                match = next(p for p in prs if p["name"] == close[0])
            else:
                subs = [p for p in prs if raw_hint.lower() in p["name"].lower() or hint_en.lower() in p["name"].lower()]
                match = subs[0] if subs else prs[0]
        creds = await tmdb_get(f"/person/{match['id']}/movie_credits", language="en-US")
        movies = creds.get("cast", [])
        movies.sort(key=lambda m: m.get("popularity",0), reverse=True)
    if not movies:
        await update.message.reply_text(M["no_results"][lang])
        return MENU
    ctx.user_data["last_query"] = {"movies": movies, "type": "actor", "index": 0}
    await send_movie(update, movies[0], lang)
    return ACTOR_MAIN
async def actor_custom_choice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    ctx.user_data["actor"] = None
    return await actor_custom(update, ctx)

async def actor_custom_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["actor"] = update.message.text.strip()
    return await actor_custom(update, ctx)

# Year
@with_logging("year_menu")
async def year_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lang = ctx.user_data["lang"]
    kb = [
        [Btn(M["buttons"]["dont_know"][lang], callback_data="year_none")],
        [Btn(M["buttons"]["back"][lang],     callback_data="go_menu")]
    ]
    await update.callback_query.edit_message_text(M["enter_year"][lang], reply_markup=Mk(kb))
    return YEAR

async def year_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try: await update.callback_query.answer()
    except BadRequest: pass
    lang = ctx.user_data["lang"]
    await update.callback_query.message.chat.send_action(ChatAction.TYPING)
    movies = await discover(sort_by="popularity.desc")
    ctx.user_data["last_query"] = {"movies": movies, "type": "year_none", "index": 0}
    await send_movie(update, movies[0], lang)
    return YEAR

@with_logging("year_text")
async def year_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lang = ctx.user_data["lang"]
    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        yr = int(update.message.text)
        movies = await discover(primary_release_year=yr, sort_by="popularity.desc")
    except:
        movies = await discover(sort_by="popularity.desc")
    if not movies:
        await update.message.reply_text(M["no_results"][lang])
        return MENU
    ctx.user_data["last_query"] = {"movies": movies, "type": "year", "index": 0}
    await send_movie(update, movies[0], lang)
    return YEAR

# Top10
@with_logging("top10")
async def top10(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try: await update.callback_query.answer()
    except BadRequest: pass
    lang   = ctx.user_data["lang"]
    params = {"sort_by":"vote_average.desc","vote_count_gte":1000}
    page1  = await discover(page=1, **params)
    top10  = page1[:10]
    ctx.user_data["last_query"] = {"type":"top10","params":params,"page":1,"movies":top10}
    await send_top10(update, top10, lang)
    return MENU
# ───────────────────────── Custom‐search─────────────────────────
@with_logging("start_custom")
async def start_custom(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    lang = ctx.user_data["lang"]
    await update.callback_query.edit_message_text(
        "Кто один из актёров? Введите имя или выберите «Не знаю»:",
        reply_markup=Mk([
            [Btn("Не знаю", callback_data="actor_custom_none")],
            [Btn("⬅️ Домой", callback_data="go_menu")]
        ])
    )
    return ACTOR_CUSTOM

# 2) Приём актёра → спрашиваем жанр
async def actor_custom(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lang = ctx.user_data["lang"]
    # текст вопроса из messages.json
    text = M["select_genre"][lang]

    opts = GENRES[lang] + [(M["buttons"]["dont_know"][lang], "none")]
    kb   = Mk([[Btn(name, callback_data=f"genre_{gid}")] for name, gid in opts])

    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.reply_text(text, reply_markup=kb)
    else:
        await update.message.reply_text(text, reply_markup=kb)

    return CUSTOM_GENRE

async def custom_genre(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    gid = update.callback_query.data.split("_")[1]
    ctx.user_data["genre"] = None if gid == "none" else int(gid)

    lang = ctx.user_data["lang"]
    await update.callback_query.edit_message_text(
        M["enter_year"][lang],
        reply_markup=Mk([
            [Btn(M["buttons"]["dont_know"][lang], callback_data="year_none")]
        ])
    )
    return CUSTOM_YEAR

async def custom_year_none(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    lang = ctx.user_data["lang"]
    ctx.user_data["year"] = None
    kb = Mk([
        [Btn(M["buttons"]["dont_know"][lang], callback_data="desc_none")],
        [Btn(M["buttons"]["home"][lang], callback_data="go_menu")],
    ])

    await update.callback_query.edit_message_text(
        M["describe_movie"][lang],
        reply_markup=kb
    )
    return CUSTOM_DESC

async def custom_year_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lang = ctx.user_data["lang"]
    text = update.message.text.strip()

    if not text.isdigit():
        await update.message.reply_text(
            M["enter_year"][lang],
            reply_markup=Mk([
                [Btn(M["buttons"]["dont_know"][lang], callback_data="year_none")]
            ])
        )
        return CUSTOM_YEAR

    ctx.user_data["year"] = int(text)
    kb = Mk([
        [Btn(M["buttons"]["dont_know"][lang], callback_data="desc_none")],
        [Btn(M["buttons"]["home"][lang],     callback_data="go_menu")],
    ])
    await update.message.reply_text(M["describe_movie"][lang], reply_markup=kb)
    return CUSTOM_DESC

async def custom_desc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.callback_query and update.callback_query.data == "desc_none":
        await update.callback_query.answer()
        desc = ""
    else:
        desc = update.message.text.strip()
    actor = ctx.user_data.get("actor")
    genre = ctx.user_data.get("genre")
    year  = ctx.user_data.get("year")
    lang  = ctx.user_data["lang"]
    if desc == "" and not any([actor, genre, year]):
        await (update.callback_query.message if update.callback_query else update.message) \
            .reply_text("Пожалуйста, введите описание или выберите жанр/год/актёра.")
        return CUSTOM_DESC
    if desc:
        if any([actor, genre, year]):
            params = {"sort_by":"popularity.desc"}
            if genre: params["with_genres"] = genre
            if year:  params["primary_release_year"] = year
            if actor:
                prs = (await tmdb_get("/search/person", query=actor, language="en-US")) \
                      .get("results", [])
                if prs: params["with_cast"] = prs[0]["id"]
            movies = await discover(page=1, **params)
            if movies:
                sem = await super_search_async(desc, k=len(movies))
                ids = {m["id"] for m in movies}
                filtered = [m for m in sem if m["id"] in ids] or movies
            else:
                filtered = []
        else:
            filtered = await search_by_description(desc, lang)

    else:
        params = {"sort_by":"popularity.desc"}
        if genre: params["with_genres"] = genre
        if year:  params["primary_release_year"] = year
        if actor:
            prs = (await tmdb_get("/search/person", query=actor, language="en-US")) \
                  .get("results", [])
            if prs: params["with_cast"] = prs[0]["id"]
        filtered = await discover(page=1, **params)
    if not filtered:
        params = {"sort_by": "popularity.desc"}
        if ctx.user_data.get("genre"):
            params["with_genres"] = ctx.user_data["genre"]
        if ctx.user_data.get("year"):
            params["primary_release_year"] = ctx.user_data["year"]
        if ctx.user_data.get("actor"):
            prs = (await tmdb_get(
                "/search/person",
                query=ctx.user_data["actor"],
                language="en-US"
            )).get("results", [])
            if prs:
                params["with_cast"] = prs[0]["id"]
        fallback = await discover(page=1, **params)

        if fallback:
            filtered = fallback
        else:
            await (update.callback_query.message if update.callback_query else update.message) \
                .reply_text(M["no_results"][lang])
            return MENU
    ctx.user_data["last_query"] = {"movies": filtered, "type": "custom", "index": 0}
    await send_movie(update, filtered[0], lang)
    return CUSTOM_DESC
# Feedback
@with_logging("rating_cb")
async def rating_cb(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await update.callback_query.answer()
    except BadRequest:
        pass

    lang = ctx.user_data["lang"]
    code = update.callback_query.data.split("_")[1]  # "like", "dislike" или "comment"
    if code == "comment":
        await update.callback_query.edit_message_text(M["ask_comment"][lang])
        return COMMENT
    cur.execute(
        "INSERT INTO feedback(user_id, lang, rating) VALUES (?,?,?)",
        (update.effective_user.id, lang, code)
    )
    con.commit()

    await update.callback_query.edit_message_text(LABELS[lang]["thanks"])
    await update.effective_chat.send_message(M["ask_age"][lang])
    return AGE

async def comment_save(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lang = ctx.user_data["lang"]
    raw = update.message.text.strip()
    cur.execute(
        "INSERT INTO feedback(user_id, lang, rating, comment) VALUES (?,?,?,?)",
        (update.effective_user.id, lang, "comment", raw)
    )
    con.commit()

    await update.message.reply_text(LABELS[lang]["thanks"])
    await update.message.reply_text(M["ask_age"][lang])
    return AGE

# Save age
async def save_age(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    lang = ctx.user_data["lang"]
    text = update.message.text.strip()
    if not text.isdigit():
        await update.message.reply_text(
            M["age_invalid"][lang]
        )
        return AGE
    age = int(text)
    cur.execute(
        "UPDATE feedback SET age = ? "
        "WHERE rowid = (SELECT MAX(rowid) FROM feedback WHERE user_id = ?)",
        (age, update.effective_user.id)
    )
    con.commit()
    await update.message.reply_text(
        M["age_saved"][lang]
    )
    return ConversationHandler.END

# Repeat
async def repeat_last(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        await update.callback_query.answer()
    except BadRequest:
        pass

    lang = ctx.user_data["lang"]
    q    = ctx.user_data.get("last_query", {})
    movies = q.get("movies", [])
    if not movies:
        await update.callback_query.edit_message_text(M["no_results"][lang])
        return MENU

    if q.get("type") in ("genre", "actor", "year"):
        idx = q.get("index", 0) + 1
        if idx < len(movies):
            q["index"] = idx
            await send_movie(update, movies[idx], lang)
            return {"genre": GENRE, "actor": ACTOR_MAIN, "year": YEAR}[q["type"]]
        else:
            await update.callback_query.edit_message_text(M["no_results"][lang])
            return MENU
    if q.get("type") == "top10":
        params  = q["params"]
        current = q.get("page", 1)
        new_page = current + 1
        page_movies = await discover(page=new_page, **params)
        if not page_movies:
            await update.callback_query.edit_message_text(M["no_results"][lang])
            return MENU

        top_list      = page_movies[:10]
        q["page"]     = new_page
        q["movies"]   = top_list

        await send_top10(update, top_list, lang)
        return MENU

    if q.get("type") == "custom":
        idx = q.get("index", 0) + 1
        if idx < len(movies):
            q["index"] = idx
            await send_movie(update, movies[idx], lang)
            return CUSTOM_DESC
        else:
            await update.callback_query.edit_message_text(M["no_results"][lang])
            return MENU

# Cancel
async def cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bye!")
    return ConversationHandler.END



# ───────────────────────── Main ─────────────────────────
if __name__ == "__main__":
    app = ApplicationBuilder().token(config.TELEGRAM_TOKEN).build()

    # хендлеры для «повтора» и «завершения»
    repeat_h = CallbackQueryHandler(repeat_last, pattern="^repeat_last$")
    finish_h = CallbackQueryHandler(finish_cb,   pattern="^finish$")

    conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CallbackQueryHandler(start_custom, pattern="^custom_search$")
        ],
        states={
            LANG: [
                CallbackQueryHandler(set_lang, pattern="^lang_"),
                repeat_h
            ],
            MENU: [
                repeat_h,
                CallbackQueryHandler(menu_router)
            ],
            GENRE: [
                repeat_h,
                CallbackQueryHandler(genre_pick, pattern="^genre_"),
                finish_h,
                CallbackQueryHandler(menu_router)
            ],
            ACTOR_MAIN: [
                repeat_h,
                CallbackQueryHandler(finish_cb, pattern="^finish$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, actor_main_text),
                CallbackQueryHandler(menu_router)
            ],
            YEAR: [
                repeat_h,
                CallbackQueryHandler(year_button, pattern="^year_none$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, year_text),
                finish_h,
                CallbackQueryHandler(menu_router)
            ],
            ACTOR_CUSTOM: [
                CallbackQueryHandler(actor_custom_choice, pattern="^actor_custom_none$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, actor_custom_text),
            ],
            CUSTOM_GENRE: [
                CallbackQueryHandler(custom_genre, pattern="^genre_")
            ],
            CUSTOM_YEAR: [
                CallbackQueryHandler(custom_year_none, pattern="^year_none$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, custom_year_text),
            ],
            CUSTOM_DESC: [
                repeat_h,
                finish_h,
                CallbackQueryHandler(custom_desc, pattern="^desc_none$"),
                CallbackQueryHandler(menu_router, pattern="^go_menu$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, custom_desc),
            ],
            RATING: [
                repeat_h,
                CallbackQueryHandler(rating_cb, pattern="^rate_"),
                CallbackQueryHandler(menu_router)
            ],
            COMMENT: [
                repeat_h,
                MessageHandler(filters.TEXT & ~filters.COMMAND, comment_save),
                CallbackQueryHandler(menu_router)
            ],
            AGE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, save_age),
                CallbackQueryHandler(menu_router)
            ],
        },
        fallbacks=[
            CommandHandler("cancel", cancel)
        ],
    )

    app.add_handler(conv)
    log.info("Bot starting (polling)…")
    app.run_polling()


