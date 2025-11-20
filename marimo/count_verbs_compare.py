import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import dspy
    from insl.correction import init_spelling_corrector, tokenize_text, convert_to_german_quotes
    from insl.parser import count_verbs, highlight_verbs, init_parser
    from sacremoses import MosesDetokenizer

    return (
        mo,
        dspy,
        init_spelling_corrector,
        tokenize_text,
        convert_to_german_quotes,
        count_verbs,
        highlight_verbs,
        init_parser,
        MosesDetokenizer,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # German Text Verb Counter - Comparison

        Compare verb counts between two texts (e.g., original vs. corrected).
        """
    )
    return


@app.cell
def __(
    init_spelling_corrector,
    init_parser,
    mo,
):
    # Initialize models with marimo caching
    @mo.cache
    def load_models():
        # Load parser (always same)
        nlp = init_parser()
        # Build corrector with openrouter
        corrector = init_spelling_corrector(provider="openrouter")
        return corrector, nlp

    corrector, nlp = load_models()

    mo.md("Models loaded - using **OpenRouter** for corrections!")

    return corrector, nlp, load_models


@app.cell
def __(mo):
    # Create tabs for two text inputs
    text_input_1 = mo.ui.text_area(
        value="""Herr Jakob beobachtete den schönen Brunnen.
Die Boote fuhren los.
Die Boote fahrten im runden Brunnen.""",
        label="Text 1:",
        full_width=True,
        rows=5,
    )

    text_input_2 = mo.ui.text_area(
        value="""Herr Jakob beobachtete den schönen Brunnen.
Die Männer ließen ihre Boote zu großem
Wasser losfahren.
Die Boote fuhren im runden Brunnen.""",
        label="Text 2:",
        full_width=True,
        rows=5,
    )

    tabs = mo.ui.tabs({
        "Text 1": text_input_1,
        "Text 2": text_input_2,
    })

    tabs
    return text_input_1, text_input_2, tabs


@app.cell
def __(mo):
    process_btn = mo.ui.run_button(label="Compare Texts")
    process_btn
    return (process_btn,)


@app.cell
def __(
    mo,
    dspy,
    count_verbs,
    highlight_verbs,
    corrector,
    nlp,
    text_input_1,
    text_input_2,
    process_btn,
    tokenize_text,
    convert_to_german_quotes,
    MosesDetokenizer,
):
    mo.stop(not process_btn.value)

    # Helper to get cost from DSPy history
    def get_llm_cost():
        cost = None
        tokens = None
        if dspy.settings.lm and hasattr(dspy.settings.lm, 'history') and dspy.settings.lm.history:
            last_call = dspy.settings.lm.history[-1]
            if 'usage' in last_call:
                usage = last_call['usage']
                tokens = {
                    'prompt': usage.get('prompt_tokens', 0),
                    'completion': usage.get('completion_tokens', 0),
                    'total': usage.get('total_tokens', 0),
                }
            if 'cost' in last_call:
                cost = last_call['cost']
        return cost, tokens

    total_cost = 0.0
    total_tokens = {'prompt': 0, 'completion': 0, 'total': 0}

    # Process both texts (4 stages each: Tokenize, LLM, POS, Count)
    with mo.status.progress_bar(total=8, title="Processing texts...") as progress:
        # Text 1
        progress.update(title="Text 1: Tokenizing...")
        cleaned_1 = text_input_1.value.replace("\n", " ").strip()
        tokens_1 = tokenize_text(cleaned_1)
        progress.update(increment=1)

        progress.update(title="Text 1: LLM Correction...")
        corrected_tokens_1 = corrector(tokens=tokens_1).corrected
        cost_1, tokens_1_usage = get_llm_cost()
        if cost_1:
            total_cost += cost_1
        if tokens_1_usage:
            total_tokens['prompt'] += tokens_1_usage['prompt']
            total_tokens['completion'] += tokens_1_usage['completion']
            total_tokens['total'] += tokens_1_usage['total']
        detokenizer = MosesDetokenizer(lang="de")
        corrected_1 = convert_to_german_quotes(detokenizer.detokenize(corrected_tokens_1))
        progress.update(increment=1)

        progress.update(title="Text 1: POS Tagging...")
        doc_1 = nlp(corrected_1)
        progress.update(increment=1)

        progress.update(title="Text 1: Counting Verbs...")
        verb_count_1, verbs_1 = count_verbs(doc_1, include_aux=True)
        highlighted_1 = highlight_verbs(doc_1, include_aux=True)
        progress.update(increment=1)

        result_1 = {
            "corrected": corrected_1,
            "verb_count": verb_count_1,
            "verbs": verbs_1,
            "highlighted": highlighted_1,
            "main_verbs": sum(1 for v in verbs_1 if v.upos == "VERB"),
            "aux_verbs": sum(1 for v in verbs_1 if v.upos == "AUX"),
        }

        # Text 2
        progress.update(title="Text 2: Tokenizing...")
        cleaned_2 = text_input_2.value.replace("\n", " ").strip()
        tokens_2 = tokenize_text(cleaned_2)
        progress.update(increment=1)

        progress.update(title="Text 2: LLM Correction...")
        corrected_tokens_2 = corrector(tokens=tokens_2).corrected
        cost_2, tokens_2_usage = get_llm_cost()
        if cost_2:
            total_cost += cost_2
        if tokens_2_usage:
            total_tokens['prompt'] += tokens_2_usage['prompt']
            total_tokens['completion'] += tokens_2_usage['completion']
            total_tokens['total'] += tokens_2_usage['total']
        corrected_2 = convert_to_german_quotes(detokenizer.detokenize(corrected_tokens_2))
        progress.update(increment=1)

        progress.update(title="Text 2: POS Tagging...")
        doc_2 = nlp(corrected_2)
        progress.update(increment=1)

        progress.update(title="Text 2: Counting Verbs...")
        verb_count_2, verbs_2 = count_verbs(doc_2, include_aux=True)
        highlighted_2 = highlight_verbs(doc_2, include_aux=True)
        progress.update(increment=1)

        result_2 = {
            "corrected": corrected_2,
            "verb_count": verb_count_2,
            "verbs": verbs_2,
            "highlighted": highlighted_2,
            "main_verbs": sum(1 for v in verbs_2 if v.upos == "VERB"),
            "aux_verbs": sum(1 for v in verbs_2 if v.upos == "AUX"),
        }

    # Calculate differences
    diff_total = result_2["verb_count"] - result_1["verb_count"]
    diff_main = result_2["main_verbs"] - result_1["main_verbs"]
    diff_aux = result_2["aux_verbs"] - result_1["aux_verbs"]

    def format_diff(diff):
        if diff > 0:
            return f"+{diff}"
        return str(diff)

    # Build comparison display
    comparison = mo.vstack([
        mo.md("## Comparison Results"),

        # Difference summary
        mo.md("### Difference (Text 2 - Text 1)"),
        mo.hstack([
            mo.stat(
                label="Total Verbs",
                value=format_diff(diff_total),
                caption="difference",
            ),
            mo.stat(
                label="Main Verbs",
                value=format_diff(diff_main),
                caption="difference",
            ),
            mo.stat(
                label="Auxiliaries",
                value=format_diff(diff_aux),
                caption="difference",
            ),
        ]),

        mo.md("---"),

        # Text 1 results
        mo.md("### Text 1 Results"),
        mo.hstack([
            mo.stat(label="Total Verbs", value=result_1["verb_count"]),
            mo.stat(label="Main Verbs", value=result_1["main_verbs"]),
            mo.stat(label="Auxiliaries", value=result_1["aux_verbs"]),
        ]),
        mo.md(f'<p style="font-size: 1.1em; line-height: 1.6;">{result_1["highlighted"]}</p>'),

        mo.md("---"),

        # Text 2 results
        mo.md("### Text 2 Results"),
        mo.hstack([
            mo.stat(label="Total Verbs", value=result_2["verb_count"]),
            mo.stat(label="Main Verbs", value=result_2["main_verbs"]),
            mo.stat(label="Auxiliaries", value=result_2["aux_verbs"]),
        ]),
        mo.md(f'<p style="font-size: 1.1em; line-height: 1.6;">{result_2["highlighted"]}</p>'),

        mo.md("""
<small>
<span style="color: #0066cc;">■</span> Main verbs (VERB) &nbsp;&nbsp;
<span style="color: #ff8c00;">■</span> Auxiliary verbs (AUX)
</small>
"""),
    ])

    # Build cost info string
    cost_info = ""
    if total_cost > 0:
        cost_info = f"Total Cost: ${total_cost:.6f}"
    if total_tokens['total'] > 0:
        tokens_str = f"Total Tokens: {total_tokens['total']} (prompt: {total_tokens['prompt']}, completion: {total_tokens['completion']})"
        cost_info = f"{cost_info} | {tokens_str}" if cost_info else tokens_str

    if cost_info:
        comparison = mo.vstack([comparison, mo.md(f"<small>{cost_info}</small>")])

    comparison
    return result_1, result_2, diff_total, diff_main, diff_aux, comparison


if __name__ == "__main__":
    app.run()
