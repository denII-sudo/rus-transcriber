"""Голосовой разбор пакета на ноутбуке, без экрана и без Claude.

Для каждого места без ответа: озвучивает вопрос, проигрывает клип, читает варианты и рекомендацию,
после сигнала записывает ответ (по умолчанию 5 с), распознаёт GigaAM и сопоставляет с вариантами.
Понимает: «первый»/«второй»/«1»…, имя или слова из варианта, «да» / «как предложено» (рекомендация),
«повтори», «пропусти», «оставь человеку». Остальное сохраняется как свой ответ — его разберёт Claude.
Ответы пишутся в answers.json пакета с source=local.

    python voice_review.py <...transcript.json> --status <файл статуса> [--seconds 5] [--ids r3,r7]
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402,F401
import tmpfiles  # noqa: E402
import review_pack as P  # noqa: E402
import voice_io as V  # noqa: E402

NUM = ["первый", "второй", "третий", "четвёртый"]


class Status:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.data = {"state": "running", "kind": "voice_review", "stage": "старт", "progress": 0,
                     "started_ts": round(time.time(), 1), "stage_started_ts": round(time.time(), 1)}

    def set(self, **kw):
        if "stage" in kw and kw["stage"] != self.data.get("stage"):
            kw["stage_started_ts"] = round(time.time(), 1)
        self.data.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"[{self.data['updated']}] {self.data.get('stage')}", file=sys.stderr, flush=True)
        if self.path:
            P._write(self.path, self.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--status")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    st = Status(a.status)
    try:
        man = P.manifest(a.json_path)
        if not man:
            raise RuntimeError("пакет разбора не собран")
        pd = P.pack_dir(a.json_path)
        store = P._read(pd / "answers.json", {"answers": {}})
        only = {x.strip() for x in a.ids.split(",") if x.strip()}
        todo = [it for it in man["items"] if (it["id"] in only if only else it["id"] not in store["answers"])]
        st.set(stage="загрузка распознавания", total=len(todo))
        P._asr()
        _, vname = V.voice()
        st.set(stage="разбор голосом", voice=vname, done=f"0/{len(todo)}")
        V.speak(f"Голосовой разбор. Мест: {len(todo)}. Отвечайте после сигнала: номер варианта, "
                f"«да» — согласиться с рекомендацией, «повтори», «пропусти» или «оставь человеку».")
        results = []
        tmp = tmpfiles.scratch("vr_")
        for n, it in enumerate(todo, 1):
            st.set(stage="разбор голосом", item=it["id"], done=f"{n - 1}/{len(todo)}",
                   progress=int(100 * (n - 1) / max(1, len(todo))))
            ans = None
            for attempt in range(3):
                V.speak(f"Место {n} из {len(todo)}. {it['question']}")
                V.play(pd / it["clip"])
                parts = [f"{NUM[k] if k < 4 else k + 1}: {o['label']}" for k, o in enumerate(it["options"])]
                rec = it.get("recommended")
                V.speak("Варианты. " + ". ".join(parts) + "."
                        + (f" Рекомендую {NUM[rec]}." if isinstance(rec, int) and rec < 4 else ""))
                V.beep()
                wav = pd / "voice" / f"{it['id']}-local-{int(time.time())}.wav"
                V.record(wav, a.seconds)
                text = P.recognize_here([wav]).get(str(wav), "")
                m = P.match(text, it)
                if m["kind"] == "repeat":
                    continue
                if not text and attempt == 0:
                    V.speak("Не расслышал. Ещё раз.")
                    continue
                ans = {"id": it["id"], "version": man["version"], "text": text, "audio": str(wav), "heard": True,
                       **({"kind": m["kind"]} if m["kind"] != "option" else {"option": m["option"]})}
                break
            if ans is None:
                V.speak("Пропускаю.")
                continue
            rep = P.add_answers(a.json_path, [ans], "local")
            acc = rep["accepted"][0] if rep["accepted"] else None
            if acc and acc.get("kind") in ("option", "recommended"):
                V.speak(f"Принято: {acc['answer']}.")
            elif acc and acc.get("kind") == "human":
                V.speak("Оставляю человеку.")
            elif acc and acc.get("kind") == "free":
                V.speak("Записал ваш ответ.")
            else:
                V.speak("Пропускаю.")
            results.append({"id": it["id"], "heard": text, **(acc or {})})
        V.speak("Разбор закончен.")
        st.set(state="done", stage="готово", progress=100, done=f"{len(todo)}/{len(todo)}", results=results,
               answers=str(pd / "answers.json"))
    except Exception as ex:
        import traceback
        st.set(state="error", stage="ошибка", error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc()[-1500:])
        raise


if __name__ == "__main__":
    main()
