"""End-to-end evaluation for Visual Document Q&A (full pipeline vs text-only ablation).

Run from the repository root with the real provider configured in .env:

    python /path/to/eval/run_eval.py --questions /path/to/questions.json --pdf-dir /path/to/pdfs

Offline smoke test of the harness itself (fake models, numbers are meaningless):

    python /path/to/eval/run_eval.py --questions eval/questions_smoke.json --pdf-dir eval/corpus --fake

questions.json is a list of objects:
    {"id": "q1", "pdf": "report.pdf", "category": "chart|text|table|scan",
     "question": "...", "answer": "reference answer", "pages": [3]}

Outputs (next to --out, default eval/results): results.json, answers.csv (for manual
grading), and tables.tex (rows for Tables 5.3 and 5.5 of the thesis).
"""

import argparse
import csv
import dataclasses
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.getcwd())

import logic  # noqa: E402  (repository module)

JUDGE_PROMPT = (
    "You grade answers to questions about a document.\n"
    "Question: {q}\nReference answer: {ref}\nCandidate answer: {cand}\n"
    "Reply with exactly YES if the candidate agrees with the reference in substance, "
    "otherwise reply exactly NO."
)


def cited_pages(answer: str) -> set[int]:
    pages = set()
    for m in re.finditer(r"pages?\s*((?:\d+\s*(?:,|and|&|-|–)?\s*)+)", answer, re.I):
        pages.update(int(x) for x in re.findall(r"\d+", m.group(1)))
    return pages


def setup(mode: str, root: str, fake: bool):
    """Point the pipeline at an isolated index directory for this mode."""
    overrides = dict(chroma_dir=os.path.join(root, f"chroma_{mode}"),
                     image_dir=os.path.join(root, "images"))
    if mode == "text_only":
        overrides["max_images"] = 0  # parser then returns text only
    if fake:
        overrides["vision_delay_seconds"] = 0
    logic.settings = dataclasses.replace(logic.settings, **overrides)


def install_fakes():
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.language_models import FakeListChatModel
    vision = FakeListChatModel(responses=["A chart with several bars."])
    chat = FakeListChatModel(responses=["Fake answer (page 2)."])
    emb = DeterministicFakeEmbedding(size=64)
    logic.get_vision_model = lambda: vision
    logic.get_chat_model = lambda: chat
    logic.get_embeddings = lambda: emb


def judge(q, ref, cand, fake):
    if fake:
        return None
    msg = JUDGE_PROMPT.format(q=q, ref=ref, cand=cand)
    out = logic.get_chat_model().invoke(msg).text.strip().upper()
    return out.startswith("YES")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    questions = json.load(open(args.questions, encoding="utf-8"))
    if args.fake:
        install_fakes()

    vision_calls = defaultdict(int)
    orig_summ = logic._summarize_image

    def counting(image):
        vision_calls["n"] += 1
        return orig_summ(image)

    logic._summarize_image = counting

    results, rows, cost, unindexable = {}, [], [], []
    for mode in ("full", "text_only"):
        setup(mode, args.out, args.fake)
        k = logic.settings.retrieval_k
        docs = {}
        for pdf in sorted({q["pdf"] for q in questions}):
            path = os.path.join(args.pdf_dir, pdf)
            vision_calls["n"] = 0
            try:
                t0 = time.perf_counter(); d = logic.index_pdf(path); t1 = time.perf_counter() - t0
            except ValueError as e:  # e.g. scanned PDF with no text layer in text-only mode
                print(f"[{mode}] {pdf}: not indexable ({e})")
                docs[pdf] = None
                unindexable.append(dict(mode=mode, pdf=pdf, reason=str(e)))
                continue
            calls = vision_calls["n"]
            t0 = time.perf_counter(); logic.index_pdf(path); t2 = time.perf_counter() - t0
            docs[pdf] = d
            if mode == "full":
                cost.append(dict(pdf=pdf, pages=d.page_count, vision_calls=calls,
                                 first_s=round(t1, 2), cached_s=round(t2, 3),
                                 warnings=len(d.warnings)))
        per_cat = defaultdict(lambda: dict(n=0, hit=0, correct=0, cite=0.0, judged=0))
        for q in questions:
            d = docs[q["pdf"]]
            if d is None:  # nothing could be indexed: counts as a miss
                c = per_cat[q["category"]]
                c["n"] += 1; c["judged"] += 1
                rows.append(dict(mode=mode, id=q["id"], category=q["category"], hit=0,
                                 cite_precision=0.0, judge=False, question=q["question"],
                                 reference=q["answer"], answer="(document not indexable)"))
                continue
            store = logic._open_store(d.collection_name)
            retrieved = store.similarity_search(q["question"], k=k)
            hit = bool({r.metadata.get("page") for r in retrieved} & set(q["pages"]))
            res = logic.answer_question(d, q["question"])
            cp = cited_pages(res.answer)
            prec = len(cp & set(q["pages"])) / len(cp) if cp else 0.0
            ok = judge(q["question"], q["answer"], res.answer, args.fake)
            c = per_cat[q["category"]]
            c["n"] += 1; c["hit"] += hit; c["cite"] += prec
            if ok is not None:
                c["judged"] += 1; c["correct"] += int(ok)
            rows.append(dict(mode=mode, id=q["id"], category=q["category"], hit=int(hit),
                             cite_precision=round(prec, 3), judge=ok,
                             question=q["question"], reference=q["answer"], answer=res.answer))
        results[mode] = {cat: dict(questions=c["n"], hit_at_k=c["hit"] / c["n"],
                                   accuracy=(c["correct"] / c["judged"]) if c["judged"] else None,
                                   cite_precision=c["cite"] / c["n"])
                         for cat, c in per_cat.items()}

    json.dump(dict(retrieval_k=logic.settings.retrieval_k, quality=results, cost=cost, unindexable=unindexable,
                   fake=args.fake), open(os.path.join(args.out, "results.json"), "w", encoding="utf-8"), indent=2)
    with open(os.path.join(args.out, "answers.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def fmt(x):
        return "n/a" if x is None else f"{x:.2f}"
    names = {"full": "Full pipeline", "text_only": "Text-only ablation"}
    cats = {"text": "Text", "chart": "Chart", "table": "Table", "scan": "Scanned page"}
    with open(os.path.join(args.out, "tables.tex"), "w", encoding="utf-8") as f:
        f.write("% Table 5.3 rows: Configuration & Category & Hit@k & Accuracy & CitePrec\n")
        for mode in ("full", "text_only"):
            for cat, label in cats.items():
                r = results[mode].get(cat)
                if r:
                    f.write(f"{names[mode]} & {label} & {fmt(r['hit_at_k'])} & "
                            f"{fmt(r['accuracy'])} & {fmt(r['cite_precision'])} \\\\\n")
        f.write("% Table 5.5 rows: Document & Pages & Vision calls & First load (s) & Cached (s)\n")
        for c in cost:
            f.write(f"{c['pdf']} & {c['pages']} & {c['vision_calls']} & "
                    f"{c['first_s']} & {c['cached_s']} \\\\\n")
    print(open(os.path.join(args.out, "tables.tex"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
