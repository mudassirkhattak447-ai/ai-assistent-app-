# 📄 ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) friendliness and tells you exactly how to improve it. Powered by Google's **Gemini Flash** model.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- **Overall ATS score (0–100)** with a breakdown across five categories:
  keywords, content impact, structure, formatting/ATS compatibility, readability
- **Prioritised improvements** (high / medium / low) with concrete fixes and examples
- **Missing keywords** and **bullet-point rewrites** (uses only facts already in your resume)
- Optional **job description match** — paste a job posting to score against it
- **Quick checks** that run without AI: email, phone, profile links, detected sections, word count
- Downloadable Markdown report

> **Note:** The score is an AI-generated estimate. It is not the output of any real employer's ATS, and different companies' systems behave differently. Use it as a guide, not a guarantee.

## How it works

1. The app extracts text from your file (`pypdf` for PDF, `python-docx` for DOCX).
2. The text (plus optional job description) is sent to Gemini with a strict grading prompt, asking for JSON output.
3. The response is validated and cleaned. The **overall score is computed in code** as a weighted average of the category scores, so it is consistent and not just a number the model made up.
4. Results are displayed in tabs.

If the file has no extractable text (e.g. a scanned image), the app tells you — a real ATS couldn't read it either.

## Quick start (local)

**Requirements:** Python 3.10+ and a free Gemini API key from <https://aistudio.google.com/apikey>.

```bash
# 1. Clone and enter the project
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key (pick ONE option)
#    Option A: paste it into the sidebar when the app runs
#    Option B: environment variable
export GEMINI_API_KEY="your-key-here"     # Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
#    Option C: Streamlit secrets file
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "your-key-here"' > .streamlit/secrets.toml

# 5. Run
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Configuration

| Setting          | Where                              | Default            | Purpose                          |
|------------------|------------------------------------|--------------------|----------------------------------|
| `GEMINI_API_KEY` | Sidebar, env var, or Streamlit secrets | —              | Your Gemini API key (required)   |
| `GEMINI_MODEL`   | Sidebar, env var, or Streamlit secrets | `gemini-2.5-flash` | Which Gemini model to use     |

Google retires older model names from time to time. If you see "model not found", change the model in the sidebar to a current Flash model listed at <https://ai.google.dev/gemini-api/docs/models> (for example `gemini-flash-latest`).

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (see below). **Never commit your API key.**
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**. Your app gets a public `*.streamlit.app` URL.

## Project structure

```
.
├── app.py              # Streamlit app (UI + analysis logic)
├── requirements.txt    # Python dependencies
├── README.md
└── .gitignore          # keeps secrets and venv out of Git
```

Recommended `.gitignore`:

```
.venv/
__pycache__/
.streamlit/secrets.toml
.env
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "API key looks invalid" | Re-copy the key from Google AI Studio; check for extra spaces |
| "Rate limit or quota reached" | The free tier has per-minute and daily limits — wait and retry |
| "Model name was not found" | Update the model name in the sidebar |
| "Could not read enough text" | Your PDF is probably an image/scan. Export a text-based PDF or use DOCX |
| `ModuleNotFoundError` | Activate your virtual environment and run `pip install -r requirements.txt` |

## Privacy

Resume text is sent to Google's Gemini API for analysis. This app does not save uploaded files or results to disk. Check Google's API data-use terms before uploading sensitive documents, and remove personal details you don't want shared.

## License

MIT — use it however you like.
