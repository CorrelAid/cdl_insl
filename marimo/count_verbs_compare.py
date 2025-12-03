import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import dspy
    import plotly.graph_objects as go
    from insl.correction import init_spelling_corrector, tokenize_text, convert_to_german_quotes
    from insl.parser import count_verbs, highlight_verbs, init_parser
    from sacremoses import MosesDetokenizer

    return (
        mo,
        dspy,
        go,
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
        # InSL Text Analysis

        Analyze and compare metrics across multiple German texts.

        ## How it works

        1. **Spelling correction**: Each text is corrected word-by-word using [mistralai/mistral-small-3.2-24b-instruct](https://openrouter.ai/mistralai/mistral-small-3.2-24b-instruct) via OpenRouter ([view prompt](https://github.com/CorrelAid/cdl_insl/blob/main/src/insl/correction.py)). 
            - This is done to ensure stanza models (trained on correct text) are working as expected.
            - This step incurs API costs, displayed at the bottom of the results.
        2. **POS tagging**: Parts of speech are tagged using the [Stanza](https://stanfordnlp.github.io/stanza/) NLP library
        3. **Metric extraction**: Total words, verbs (VERB), and auxiliaries (AUX) are counted and compared

        Enter 1-3 texts below to analyze individually or compare across versions.
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

    with mo.status.spinner(title="Loading Stanza models..."):
        corrector, nlp = load_models()

    mo.md("✓ Stanza Models loaded - using **OpenRouter** for corrections!")

    return corrector, nlp, load_models


@app.cell
def __(mo):
    # Create three text inputs
    text_input_1 = mo.ui.text_area(
        value="""Herr Jakob beobachtete den schönen Brunnen.
Die Boote fuhren los.
Die Boote fahrten im runden Brunnen.""",
        label="Text 1 (required):",
        full_width=True,
        rows=5,
    )

    text_input_2 = mo.ui.text_area(
        value="""Herr Jakob beobachtete den schönen Brunnen.
Die Männer ließen ihre Boote zu großem
Wasser losfahren.
Die Boote fuhren im runden Brunnen.""",
        label="Text 2 (optional):",
        full_width=True,
        rows=5,
    )

    text_input_3 = mo.ui.text_area(
        value="",
        label="Text 3 (optional):",
        full_width=True,
        rows=5,
    )

    mo.vstack([text_input_1, text_input_2, text_input_3])
    return text_input_1, text_input_2, text_input_3


@app.cell
def __(mo):
    process_btn = mo.ui.run_button(label="Process Texts")
    process_btn
    return (process_btn,)


@app.cell
def __(
    mo,
    dspy,
    go,
    count_verbs,
    highlight_verbs,
    corrector,
    nlp,
    text_input_1,
    text_input_2,
    text_input_3,
    process_btn,
    tokenize_text,
    convert_to_german_quotes,
    MosesDetokenizer,
):
    # Create state to track last processed button value
    get_last_processed, set_last_processed = mo.state(0)

    # Stop if button hasn't been clicked or if we already processed this click
    current_click = process_btn.value
    if current_click == 0 or current_click == get_last_processed():
        mo.stop(True)

    # Mark this click as processed
    set_last_processed(current_click)

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

    # Determine which texts to process
    texts_to_process = []
    if text_input_1.value.strip():
        texts_to_process.append(("Text 1", text_input_1.value))
    if text_input_2.value.strip():
        texts_to_process.append(("Text 2", text_input_2.value))
    if text_input_3.value.strip():
        texts_to_process.append(("Text 3", text_input_3.value))

    if not texts_to_process:
        mo.md("⚠️ Please provide at least one text to process.")
        mo.stop()

    total_cost = 0.0
    total_tokens = {'prompt': 0, 'completion': 0, 'total': 0}

    # Process texts (4 stages each: Tokenize, LLM, POS, Count)
    results = []
    detokenizer = MosesDetokenizer(lang="de")

    with mo.status.progress_bar(total=len(texts_to_process) * 4, title="Processing texts...") as progress:
        for text_name, text_value in texts_to_process:
            # Stage 1: Tokenize and count words in original
            progress.update(title=f"{text_name}: Tokenizing...")
            cleaned = text_value.replace("\n", " ").strip()
            tokens = tokenize_text(cleaned)
            # Count words in original text (tokens that are not just punctuation)
            total_words = len([t for t in tokens if not all(c in '.,!?;:"\'()[]{}' for c in t)])
            progress.update(increment=1)

            # Stage 2: LLM Correction
            progress.update(title=f"{text_name}: LLM Correction...")
            corrected_tokens = corrector(tokens=tokens).corrected
            cost, tokens_usage = get_llm_cost()
            if cost:
                total_cost += cost
            if tokens_usage:
                total_tokens['prompt'] += tokens_usage['prompt']
                total_tokens['completion'] += tokens_usage['completion']
                total_tokens['total'] += tokens_usage['total']
            corrected = convert_to_german_quotes(detokenizer.detokenize(corrected_tokens))
            progress.update(increment=1)

            # Stage 3: POS Tagging
            progress.update(title=f"{text_name}: POS Tagging...")
            doc = nlp(corrected)
            progress.update(increment=1)

            # Stage 4: Counting Verbs
            progress.update(title=f"{text_name}: Counting Verbs...")
            verb_count, verbs = count_verbs(doc, include_aux=True)
            highlighted = highlight_verbs(doc, include_aux=True)
            progress.update(increment=1)

            results.append({
                "name": text_name,
                "corrected": corrected,
                "verb_count": verb_count,
                "verbs": verbs,
                "highlighted": highlighted,
                "main_verbs": sum(1 for v in verbs if v.upos == "VERB"),
                "aux_verbs": sum(1 for v in verbs if v.upos == "AUX"),
                "total_words": total_words,
            })

    # Build display based on number of texts
    def format_diff(diff):
        if diff > 0:
            return f"+{diff}"
        return str(diff)

    display_components = []

    if len(results) == 1:
        # Single text - just show results
        result = results[0]
        display_components.extend([
            mo.md(f"## {result['name']} Results"),
            mo.hstack([
                mo.stat(label="Total Words", value=result["total_words"]),
                mo.stat(label="Total Verbs", value=result["verb_count"]),
                mo.stat(label="Main Verbs", value=result["main_verbs"]),
                mo.stat(label="Auxiliaries", value=result["aux_verbs"]),
            ]),
            mo.md(f'<p style="font-size: 1.1em; line-height: 1.6;">{result["highlighted"]}</p>'),
            mo.md("""
<small>
<span style="color: #0066cc;">■</span> Main verbs (VERB) &nbsp;&nbsp;
<span style="color: #ff8c00;">■</span> Auxiliary verbs (AUX)
</small>
"""),
        ])
    else:
        # Multiple texts - show comparison
        display_components.append(mo.md("## Text Analysis Comparison"))

        # Add line chart for metrics
        text_names = [r['name'] for r in results]
        total_words = [r['total_words'] for r in results]
        total_verbs = [r['verb_count'] for r in results]
        main_verbs = [r['main_verbs'] for r in results]
        aux_verbs = [r['aux_verbs'] for r in results]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=text_names, y=total_words, mode='lines+markers', name='Total Words', line=dict(color='#10b981', width=3)))
        fig.add_trace(go.Scatter(x=text_names, y=total_verbs, mode='lines+markers', name='Total Verbs', line=dict(color='#8b5cf6', width=3)))
        fig.add_trace(go.Scatter(x=text_names, y=main_verbs, mode='lines+markers', name='Main Verbs', line=dict(color='#0066cc', width=2)))
        fig.add_trace(go.Scatter(x=text_names, y=aux_verbs, mode='lines+markers', name='Auxiliaries', line=dict(color='#ff8c00', width=2)))

        fig.update_layout(
            title="Text Metrics Across Texts",
            yaxis_title="Count",
            hovermode='x unified',
            template='plotly_white',
            height=400,
        )

        chart = mo.ui.plotly(fig)
        display_components.append(chart)

        # Create download button for data
        import csv
        import io

        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(['Text', 'Total Words', 'Total Verbs', 'Main Verbs', 'Auxiliaries'])
        for result in results:
            csv_writer.writerow([
                result['name'],
                result['total_words'],
                result['verb_count'],
                result['main_verbs'],
                result['aux_verbs']
            ])

        download_btn = mo.download(
            data=csv_buffer.getvalue().encode('utf-8'),
            filename="verb_counts_data.csv",
            label="Download Data as CSV",
        )

        display_components.append(download_btn)

        # Add differences between first and last text
        if len(results) >= 2:
            display_components.append(mo.md(f"### Difference ({results[-1]['name']} - {results[0]['name']})"))
            diff_words = results[-1]["total_words"] - results[0]["total_words"]
            diff_total = results[-1]["verb_count"] - results[0]["verb_count"]
            diff_main = results[-1]["main_verbs"] - results[0]["main_verbs"]
            diff_aux = results[-1]["aux_verbs"] - results[0]["aux_verbs"]
            display_components.append(mo.hstack([
                mo.stat(
                    label="Total Words (original text)",
                    value=format_diff(diff_words),
                    caption="difference",
                ),
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
            ]))
        display_components.append(mo.md("---"))
        display_components.append(mo.md("## Text Analysis"))
        for i, result in enumerate(results):
            display_components.extend([
                mo.md(f"### {result['name']}"),
                mo.hstack([
                    mo.stat(label="Total Words (original text)", value=result["total_words"]),
                    mo.stat(label="Total Verbs", value=result["verb_count"]),
                    mo.stat(label="Main Verbs", value=result["main_verbs"]),
                    mo.stat(label="Auxiliaries", value=result["aux_verbs"]),
                ]),
                mo.md(f'<p style="font-size: 1.1em; line-height: 1.6;">{result["highlighted"]}</p>'),
            ])

        display_components.append(mo.md("""
<small>
<span style="color: #0066cc;">■</span> Main verbs (VERB) &nbsp;&nbsp;
<span style="color: #ff8c00;">■</span> Auxiliary verbs (AUX)
</small>
"""))
    display_components.append(mo.md("---"))
    # Build cost info string
    cost_info = ""
    if total_cost > 0:
        cost_info = f"Total Cost for Correction with LLM: ${total_cost:.6f}"
    if total_tokens['total'] > 0:
        tokens_str = f"Total Tokens: {total_tokens['total']} (prompt: {total_tokens['prompt']}, completion: {total_tokens['completion']})"
        cost_info = f"{cost_info} | {tokens_str}" if cost_info else tokens_str

    if cost_info:
        display_components.append(mo.md(f"<small>{cost_info}</small>"))

    output = mo.vstack(display_components)
    output
    return results, output


if __name__ == "__main__":
    app.run()
