import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import dspy
    from insl.correction import correct_spelling, init_spelling_corrector, tokenize_text
    from insl.parser import count_verbs, highlight_verbs, init_parser

    return (
        mo,
        dspy,
        correct_spelling,
        init_spelling_corrector,
        tokenize_text,
        count_verbs,
        highlight_verbs,
        init_parser,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # German Text Verb Counter

        Enter German text below to get spelling correction and verb count.
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

    mo.md("✅ Models loaded - using **OpenRouter** for corrections!")

    return corrector, nlp, load_models


@app.cell
def __(mo):
    text_input = mo.ui.text_area(
        value="""Herr Jakob beobachtete den schönen Brunnen.
Die Männer ließen ihre Boote zu großen
Wasser losfahren.
Die Boote fahrten im runden Brunnen.
Die Boote segelten ins saubere Wasser.
Herr Jakob und die Männer bastelten
die großen Schienen zusammen.
Herr Jakob kehrt zurück mit einem braunen Kartong.
Herr Jakob und die Männer bauten die Schienen auf den runden Brunnen.""",
        label="Enter German text:",
        full_width=True,
        rows=5,
    )
    text_input
    return (text_input,)


@app.cell
def __(mo):
    process_btn = mo.ui.run_button(label="Process Text")
    process_btn
    return (process_btn,)


@app.cell
def __(
    mo,
    correct_spelling,
    count_verbs,
    highlight_verbs,
    corrector,
    nlp,
    text_input,
    process_btn,
    tokenize_text,
    dspy,
):
    from sacremoses import MosesDetokenizer

    mo.stop(not process_btn.value)

    # Stage tracking
    stages = ["Tokenizing", "LLM Correction", "POS Tagging", "Counting Verbs"]

    with mo.status.progress_bar(
        total=len(stages), title="Processing text..."
    ) as progress:
        # Stage 1: Tokenization
        progress.update(title=f"Stage 1/4: {stages[0]}...")
        # Remove newlines from input text
        cleaned_text = text_input.value.replace("\n", " ").strip()
        tokens = tokenize_text(cleaned_text)
        progress.update(increment=1)

        # Stage 2: LLM Spelling Correction
        progress.update(title=f"Stage 2/4: {stages[1]}...")

        # Get the corrected tokens from the corrector
        result = corrector(tokens=tokens)
        corrected_tokens = result.corrected

        # Get usage info from DSPy history
        lm_cost = None
        lm_tokens = None
        if dspy.settings.lm and hasattr(dspy.settings.lm, 'history') and dspy.settings.lm.history:
            last_call = dspy.settings.lm.history[-1]
            if 'usage' in last_call:
                usage = last_call['usage']
                lm_tokens = {
                    'prompt': usage.get('prompt_tokens', 0),
                    'completion': usage.get('completion_tokens', 0),
                    'total': usage.get('total_tokens', 0),
                }
            if 'cost' in last_call:
                lm_cost = last_call['cost']

        # Detokenize
        detokenizer = MosesDetokenizer(lang="de")
        corrected = detokenizer.detokenize(corrected_tokens)

        progress.update(increment=1)

        # Stage 3: POS Tagging
        progress.update(title=f"Stage 3/4: {stages[2]}...")
        doc = nlp(corrected)
        progress.update(increment=1)

        # Stage 4: Count Verbs
        progress.update(title=f"Stage 4/4: {stages[3]}...")
        verb_count, verbs = count_verbs(doc, include_aux=True)

        # Highlight verbs using character offsets from the parsed document
        highlighted_text = highlight_verbs(doc, include_aux=True)
        progress.update(increment=1)

    # Extract verb info for display
    verb_info = []
    for v in verbs:
        verb_type = "Main Verb" if v.upos == "VERB" else "Auxiliary"
        verb_info.append(f"- **{v.text}** ({v.lemma}) - {verb_type}")

    # Build cost info string
    cost_info = ""
    if lm_cost is not None:
        cost_info = f"Cost: ${lm_cost:.6f}"
    if lm_tokens is not None:
        tokens_str = f"Tokens: {lm_tokens['total']} (prompt: {lm_tokens['prompt']}, completion: {lm_tokens['completion']})"
        cost_info = f"{cost_info} | {tokens_str}" if cost_info else tokens_str

    result = mo.vstack(
        [
            mo.md("## Results"),
            mo.hstack(
                [
                    mo.stat(
                        label="Total Verbs", value=verb_count, caption="verbs found"
                    ),
                    mo.stat(
                        label="Main Verbs",
                        value=sum(1 for v in verbs if v.upos == "VERB"),
                        caption="VERB tags",
                    ),
                    mo.stat(
                        label="Auxiliaries",
                        value=sum(1 for v in verbs if v.upos == "AUX"),
                        caption="AUX tags",
                    ),
                ]
            ),
            mo.md("### Highlighted Text"),
            mo.md(
                f'<p style="font-size: 1.1em; line-height: 1.6;">{highlighted_text}</p>'
            ),
            mo.md("""
<small>
<span style="color: #0066cc;">■</span> Main verbs (VERB) &nbsp;&nbsp;
<span style="color: #ff8c00;">■</span> Auxiliary verbs (AUX)
</small>
"""),
            mo.md(f"<small>{cost_info}</small>") if cost_info else mo.md(""),
        ]
    )
    result
    return corrected, verb_count, verbs, highlighted_text, result, verb_info


if __name__ == "__main__":
    app.run()
