# PPL Anomaly Detector

A desktop application that measures **text perplexity (PPL)** using a causal language model to flag anomalous input. Built for **educational purposes**.

## What It Does

Perplexity quantifies how "surprising" a piece of text is to a language model. Text with unusually high perplexity may indicate anomalous or adversarial input (e.g., prompt injection attempts).

- Load any Hugging Face causal LM (default: `gpt2`)
- Enter text and a threshold
- Get the PPL score and an anomalous/normal verdict

## Screenshot

| Load a model | Enter text and calculate PPL |
|---|---|
| Modern card-based UI with model selector | Result card shows perplexity score and verdict |

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/)

## Installation

```bash
git clone https://github.com/freyWylfred/unnaturalApple.git
cd unnaturalApple
pip install torch transformers
```

## Usage

```bash
python main.py
```

1. Enter a Hugging Face model ID (e.g. `gpt2`, `distilgpt2`) and click **Load Model**.
2. Type or paste text into the input area.
3. Set a perplexity threshold (default: 100.0).
4. Click **Calculate PPL** to get the result.

## Building an Executable

A `build.bat` script is included for creating a standalone Windows `.exe` via [PyInstaller](https://pyinstaller.org/):

```bash
pip install pyinstaller
build.bat
```

The output will be in `dist/PPLDetector.exe`.

## How It Works

1. Tokenize the input text.
2. Run a forward pass through the causal LM with `labels=input_ids`.
3. Compute PPL = exp(cross-entropy loss).
4. Compare against the threshold to classify as **Normal** or **Anomalous**.

## Disclaimer

This tool is for **educational and research purposes only**. PPL-based detection is a simple heuristic and should not be used as a sole security measure.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
