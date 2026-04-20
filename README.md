# PPL Anomaly Detector

A desktop application that measures **text perplexity (PPL)** using a causal language model to flag anomalous input. Built for **educational purposes**.

## What It Does

Perplexity quantifies how "surprising" a piece of text is to a language model. Text with unusually high perplexity may indicate anomalous or adversarial input (e.g., prompt injection attempts).

- Load any Hugging Face causal LM (default: `gpt2`)
- Enter text and a threshold
- Get the PPL score and an Anomalous/Normal verdict
- **Per-line PPL distribution mode**: compute PPL for every non-empty line and visualize the results (bar chart + histogram) in a separate window

## Features

### Single-text mode

Enter any text and press **Calculate PPL** to get a single perplexity score and an Anomalous/Normal verdict based on the configured threshold.

### Per-line distribution mode (v1.1.0+)

Press **Visualize per-line PPL** to compute the PPL of each non-empty line of the input. A separate window opens with:

- A bar chart of PPL per line (bars that exceed the threshold are highlighted in red)
- A histogram of the PPL distribution
- A summary line (count, min, mean, max, and # of anomalous lines)
- A matplotlib navigation toolbar (zoom / pan / save image)

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/)
- [matplotlib](https://matplotlib.org/) (for the per-line distribution window)

See [`requirements.txt`](requirements.txt) for pinned lower bounds.

## Installation

```bash
git clone https://github.com/freyWylfred/unnaturalApple.git
cd unnaturalApple
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

1. Enter a Hugging Face model ID (e.g. `gpt2`, `distilgpt2`) and click **Load Model**.
2. Type or paste text into the input area (one sample per line if you plan to use the distribution mode).
3. Set a perplexity threshold (default: 100.0).
4. Click **Calculate PPL** for a single verdict, or **Visualize per-line PPL** to open the distribution window.

## Prebuilt Executable

A Windows x64 standalone executable is available on the [Releases](https://github.com/freyWylfred/unnaturalApple/releases) page. It bundles PyTorch, Transformers and matplotlib; model weights are downloaded from the Hugging Face Hub on first run.

## Building an Executable Yourself

```bash
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed --name unnaturalApple --collect-all transformers --collect-all torch --collect-submodules matplotlib --collect-data matplotlib main.py
```

The output will be `dist/unnaturalApple.exe` (~650 MB due to the bundled deep-learning stack).

## How It Works

1. Tokenize the input text.
2. Run a forward pass through the causal LM with `labels=input_ids`.
3. Compute PPL = exp(cross-entropy loss).
4. Compare against the threshold to classify as **Normal** or **Anomalous**.
5. In per-line mode, step 1-3 are repeated for every non-empty line, and the resulting distribution is plotted with matplotlib inside a Tk `Toplevel` window.

## Disclaimer

This tool is for **educational and research purposes only**. PPL-based detection is a simple heuristic and should not be used as a sole security measure.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

