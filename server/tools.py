"""TinyGPT tool layer — gives the chatbot real capabilities.

The model provides conversation; these tools provide real answers:
  calculator, clock/date, coin & dice, Wikipedia lookup, DuckDuckGo answers.
Routing is rule-based; the tools hit live web APIs at request time.
"""
import datetime
import random
import re
import zoneinfo

import requests

IST = zoneinfo.ZoneInfo("Asia/Kolkata")
UA = {"User-Agent": "TinyGPT-Chat/1.0 (educational project)"}
MATH_RE = re.compile(r"^[\d\s+\-*/().%]+$")
# questions about the bot itself must go to the model, not the web
SELF_RE = re.compile(
    r"\b(your name|who are you|who r u|your age|how old are you|who made you|"
    r"who created you|who built you|about yourself|yourself|who trained you|"
    r"are you (an? )?(ai|robot|bot|human)|your model)\b"
)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


# ---------------- calculator ----------------
def calc_tool(text: str):
    t = _norm(text)
    for w, s in [
        ("calculate", ""), ("what's", ""), ("what is", ""), ("compute", ""),
        ("solve", ""), ("=", ""), ("?", ""),
    ]:
        t = t.replace(w, s)
    t = t.strip()
    if not re.search(r"\d", t) or not re.search(r"[+\-*/^%]", t):
        return None
    if not MATH_RE.match(t):
        return None
    try:
        val = eval(t.replace("^", "**"), {"__builtins__": {}}, {})
    except Exception:
        return None
    return f"The answer is {val}."


# ---------------- clock / date ----------------
CLOCK_RE = re.compile(
    r"\b(what(?:'s| is)? (?:the )?time|time is it|what time|current time|"
    r"koto baje|ki bajche|ki baje|what(?:'s| is)? (?:the |today'?s )?date|"
    r"date today|what day|day is it today|aajker tarikh|aaj koto tarikh|"
    r"ajker tarikh|koto tarikh|today'?s date)\b"
)


def clock_tool(text: str):
    t = _norm(text)
    if not CLOCK_RE.search(t):
        return None
    now = datetime.datetime.now(IST)
    return (f"Right now it is {now.strftime('%I:%M %p')} "
            f"on {now.strftime('%A, %d %B %Y')} (IST).")


# ---------------- coin & dice ----------------
def dice_tool(text: str):
    t = _norm(text)
    if "flip a coin" in t or "toss a coin" in t or "coin flip" in t:
        return f"It's {random.choice(['heads', 'tails'])}!"
    if "roll" in t and ("dice" in t or "die" in t):
        n = 6
        m = re.search(r"(\d+)\s*(?:sided|side|-sided)", t)
        if m:
            n = int(m.group(1))
        return f"You rolled a {random.randint(1, n)} (d{n})."
    return None


# ---------------- Wikipedia ----------------
def _wiki_summary(title: str):
    try:
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            headers=UA, timeout=8)
        if r.status_code == 200:
            j = r.json()
            ex = j.get("extract")
            if ex:
                link = j.get("content_urls", {}).get("desktop", {}).get("page", "")
                return ex, link
    except Exception:
        pass
    return None, None


def wiki_tool(text: str):
    t = _norm(text)
    m = re.search(r"(?:who|what)(?:'s| is| was| are) (?:a |an |the )?(.+?)[?.!]*$", t)
    q = m.group(1).strip() if m else None
    if not q:
        m2 = re.search(r"(?:tell me about|search for|look up) (.+?)[?.!]*$", t)
        q = m2.group(1).strip() if m2 else None
    if not q or SELF_RE.search(t):
        return None
    title = q.replace(" ", "_")
    ex, link = _wiki_summary(title)
    if not ex:  # try opensearch to fix titles like "einstein" -> "Albert Einstein"
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "opensearch", "search": q, "limit": 1},
                headers=UA, timeout=8)
            titles = r.json()[1]
            if titles:
                ex, link = _wiki_summary(titles[0].replace(" ", "_"))
        except Exception:
            pass
    if not ex:
        return None
    parts = ex.split(". ")
    summary = ". ".join(parts[:3]).strip()
    if not summary.endswith("."):
        summary += "."
    return f"{summary} (source: Wikipedia)"


# ---------------- DuckDuckGo instant answers ----------------
def ddg_tool(text: str):
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": text, "format": "json", "no_html": 1, "skip_disambig": 1},
            headers=UA, timeout=8)
        j = r.json()
        if j.get("Answer"):
            return f"{j['Answer']} (source: DuckDuckGo)"
        if j.get("AbstractText"):
            return f"{j['AbstractText'][:400]} (source: DuckDuckGo)"
        for topic in j.get("RelatedTopics") or []:
            if isinstance(topic, dict) and topic.get("Text"):
                return f"{topic['Text'][:400]} (source: DuckDuckGo)"
    except Exception:
        pass
    return None


def route(text: str):
    """Returns (tool_name, answer) or (None, None) to let the model chat."""
    for name, fn in [("calculator", calc_tool), ("clock", clock_tool),
                     ("dice", dice_tool), ("wikipedia", wiki_tool)]:
        try:
            ans = fn(text)
        except Exception:
            ans = None
        if ans:
            return name, ans
    return None, None
