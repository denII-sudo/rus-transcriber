"""Словарь группы: участники (имя, пол), имена и названия, явные замены.

Файл — rus-transcriber.vocab.json (UTF-8). Где ищется, по порядку:
  1) путь, переданный явно (параметр vocab_path / --vocab);
  2) папка записи и до двух папок выше — так словарь лежит вместе с записями группы
     и передаётся вместе с ними или отдельно;
  3) DATA_DIR\\groups\\<группа>\\vocab.json — словарь группы (группа: параметр group, default_group, единственная).
Голосовые профили в словарь не входят (см. voices.py): они хранятся только на этом компьютере.

Формат:
{
  "group": "имя-группы",                      — к нему привязаны голосовые профили
  "participants": [{"name": "Анна", "gender": "ж", "role": "ведущая"}, ...],
  "terms": ["Вереск", "Лисицын", ...],     — имена и названия: близкие по звучанию искажения
                                                исправляются автоматически
  "replace": {"Борька": "Борис", ...},         — явные замены целого слова или фразы (всегда)
  "replace_if": [                              — замены с условием: для слов, которые бывают и обычными
    {"from": "весна", "to": "Весна",             (слово «весна» остаётся, название — после предлога)
     "after": ["в", "на", "про", ...],         — только после одного из этих слов
     "not_before": ["богу"],                   — не перед этими словами
     "case": true}                             — с учётом регистра («Весна», но не «весна»)
  ]
}
"""
import difflib
import json
import re
from pathlib import Path

FILE_NAME = "rus-transcriber.vocab.json"
WORD = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:-[А-Яа-яЁё]+)*")
_PH = str.maketrans({"ё": "и", "е": "и", "я": "и", "ы": "и", "э": "и", "й": "и", "о": "а", "ю": "у",
                     "б": "п", "в": "ф", "г": "к", "д": "т", "ж": "ш", "з": "с", "ь": None, "ъ": None})


def find(audio_path=None, explicit=None, default_file=None):
    """Возвращает (путь, данные) или (None, {}). default_file — словарь группы в DATA_DIR (paths.vocab_path)."""
    cands = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    if audio_path:
        d = Path(audio_path).resolve().parent
        for p in [d, *list(d.parents)[:2]]:
            cands.append(p / FILE_NAME)
    if default_file:
        cands.append(Path(default_file))
    for p in cands:
        if p.is_file():
            try:
                return p, json.loads(p.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as ex:
                raise RuntimeError(f"Словарь {p} повреждён: {ex}") from ex
    return None, {}


def genders(vocab):
    """{имя: 'f'|'m'} из participants."""
    out = {}
    for p in vocab.get("participants", []):
        g = str(p.get("gender", "")).strip().lower()[:1]
        if g in ("ж", "f", "w"):
            out[p["name"]] = "f"
        elif g in ("м", "m"):
            out[p["name"]] = "m"
    return out


def _ph(w):
    s = w.lower().translate(_PH)
    return re.sub(r"(.)\1+", r"\1", s)


def _low(w):
    return str(w).lower().replace("ё", "е").strip()


def _ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def _lastword(t):
    m = WORD.findall(str(t or ""))
    return m[-1] if m else None


def _firstword(t):
    m = WORD.findall(str(t or ""))
    return m[0] if m else None


class Fixer:
    def __init__(self, vocab):
        self.replace = []
        for k, v in (vocab.get("replace") or {}).items():
            k = str(k).strip()
            if k:
                body = re.sub(r"[её]", "[её]", re.escape(k), flags=re.I)
                pat = re.compile(r"(?<![А-Яа-яЁёA-Za-z])" + body + r"(?![А-Яа-яЁёA-Za-z])", re.I)
                self.replace.append((pat, str(v), k))
        self.cond = []
        for r in vocab.get("replace_if") or []:
            if r.get("from") and r.get("to"):
                self.cond.append({"from": str(r["from"]).strip(), "to": str(r["to"]),
                                  "case": bool(r.get("case")),
                                  "after": {_low(x) for x in r.get("after") or []},
                                  "not_after": {_low(x) for x in r.get("not_after") or []},
                                  "not_before": {_low(x) for x in r.get("not_before") or []}})
        terms = list(vocab.get("terms") or [])
        terms += [p["name"] for p in vocab.get("participants", []) if p.get("name")]
        self.terms = [(t, _ph(t)) for t in dict.fromkeys(terms) if " " not in t and len(t) >= 4]

    @staticmethod
    def _stem(term):
        return term[:-1] if len(term) >= 4 and term[-1].lower() in "аяоеиыйь" else term

    def _best(self, tok):
        """Лучшая замена для слова или None.

        Правила (подобраны на расшифровках 05.11 и 06.11, чтобы не трогать обычные слова):
          * падежная форма термина («Никитой», «Эврики», «Сень») не меняется, только регистр;
          * слово со строчной буквы сравнивается только целиком, от 6 букв, порог 0.92;
          * с заглавной — порог 0.85, допускается окончание до 3 букв;
          * кроме звучания сверяются и сами буквы (≥0.7), иначе «Берег» превращался бы в «Борис».
        """
        if len(tok) < 4:
            return None
        cap = tok[0].isupper()
        low = tok.lower()
        nt = _ph(tok)
        best = None
        for term, nterm in self.terms:
            stem = self._stem(term).lower()
            if low.startswith(stem) and len(low) - len(stem) <= 3:
                # та же основа: меняем только регистр, если термин — имя собственное
                if term[0].isupper() and not cap and low.startswith(term.lower()):
                    rep = term[0] + tok[1:]
                    if rep != tok:
                        return (1.0, rep)
                return None
            if nt[:1] != nterm[:1]:
                continue
            if not cap and len(tok) < 6:
                continue
            r1 = _ratio(nt, nterm) if _ratio(low, term.lower()) >= 0.7 else 0.0
            r2 = 0.0
            extra = len(tok) - len(term)
            if cap and 0 < extra <= 3 and _ratio(low[:len(term)], term.lower()) >= 0.7:
                r2 = _ratio(_ph(tok[:len(term)]), nterm)
            r = max(r1, r2)
            if r >= (0.85 if cap else 0.92) and (best is None or r > best[0]):
                best = (r, term if r1 >= r2 else term + tok[len(term):])
        if best and best[1] != tok:
            return best
        return None

    def _conditional(self, tok, prev, nxt):
        for c in self.cond:
            if not (tok == c["from"] if c["case"] else _low(tok) == _low(c["from"])):
                continue
            p, n = _low(prev or ""), _low(nxt or "")
            if c["after"] and p not in c["after"]:
                continue
            if p in c["not_after"] or n in c["not_before"]:
                continue
            return c["to"]
        return None

    def _guarded(self, tok):
        return any(_low(tok) == _low(c["from"]) for c in self.cond)

    def apply(self, text, prev=None, nxt=None):
        """-> (новый текст, [(было, стало), ...]). prev / nxt — соседние слова, когда text — одно слово
        (для замен с условием)."""
        fixes = []
        for pat, val, key in self.replace:
            def sub(m):
                fixes.append((m.group(0), val))
                return val
            text = pat.sub(sub, text)
        out, pos = [], 0
        ms = list(WORD.finditer(text))
        for k, m in enumerate(ms):
            tok = m.group(0)
            c = self._conditional(tok, ms[k - 1].group(0) if k else _lastword(prev),
                                  ms[k + 1].group(0) if k + 1 < len(ms) else _firstword(nxt)) \
                if self.cond else None
            # слово из замены с условием меняется только по условию: похожесть на термин его не трогает
            b = (1.0, c) if c else None if self._guarded(tok) else self._best(tok)
            out.append(text[pos:m.start()])
            if b:
                out.append(b[1])
                fixes.append((tok, b[1]))
            else:
                out.append(tok)
            pos = m.end()
        out.append(text[pos:])
        return "".join(out), fixes
