# Evaluation kit for the thesis

Run these on your own machine, where the Gemini key and model downloads work.

1. Clone and set up the project (`.env` with `GOOGLE_API_KEY`), then from the repo root:
   `pip install matplotlib` (only needed for the synthetic corpus).
2. Parser measurements (no model needed, already run once, results in `parser_results.json`):
   `python eval/build_corpus.py && python eval/parser_sweep.py .`
   (copy this `eval` folder into the repo, or adjust paths).
3. Model-dependent tables (Tables 5.3 and 5.5), from the repo root:
   `python eval/run_eval.py --questions eval/questions.json --pdf-dir /path/to/your/pdfs`
   `questions.json`: list of `{"id","pdf","category":"text|chart|table|scan","question","answer","pages":[n]}`.
   Write 8 to 15 questions per category from real PDFs with the reference answer and evidence page.
   Outputs: `eval/results/results.json`, `answers.csv` (review the judge verdicts by hand), `tables.tex`.
4. Paste the rows from `tables.tex` into the matching tables in `main.tex`, replace the red `\tbd` cells in
   Tables 5.1 to 5.2 (hardware, resolved model version, question counts), then reword the discussion to match the numbers.

`--fake` runs the harness with fake models to check the plumbing; its numbers are meaningless.
Free-tier note: the first load of each PDF waits 6 s per image, so keep the corpus small.
