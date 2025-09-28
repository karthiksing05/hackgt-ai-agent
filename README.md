# hackgt-ai-agent

Small CLI tool that extracts calendar-event information from a packaged packet (images + a WAV audio file) using OpenAI models. The script processes images and audio in a .tar.gz packet, asks GPT-style models to extract event titles, start/end times and concise summaries, deduplicates results, synthesizes a final title/summary, and posts the resulting event to a configured HTTP endpoint.

## Features

- Extracts text/audio cues from images and a WAV audio file using OpenAI vision/audio models.
- Clusters images and picks the best representative per cluster to avoid duplicates.
- Produces calendar-ready events (title, start, end) and a 4–5 bullet summary.
- Deduplicates similar events using an LLM.
- Posts final event JSON to a configured HTTP API endpoint.

## Quick start

Prerequisites:

- Python 3.10+ (3.11 recommended)
- A valid OpenAI API key placed in `API_KEY.txt` (the script reads this file directly)
- The Python packages listed in the Requirements section below

Setup (recommended: inside a virtualenv):

```bash
# create and activate a venv (macOS / zsh)
python3 -m venv .venv
source .venv/bin/activate

# install deps
pip install -r requirements.txt
```

If there is no `requirements.txt` in the repository, install these packages manually:

```bash
pip install openai pillow requests dateparser
```

Create `API_KEY.txt` at the repository root and paste your OpenAI API key (no newline-only whitespace):

```text
sk-...your-openai-key...
```

Input format:

- The CLI expects a single `.tar.gz` packet containing one or more images (jpg, png) and exactly one WAV (`.wav`) audio file. Files may live at the tar root or in a single extracted folder.

Example packet contents:

```
photo_0000001.jpg
photo_0000002.jpg
audio_buffer.wav
```

## Usage

Run the main script with the packet path and a user id (the id is used when posting to the API):

```bash
python ai.py path/to/packet.tar.gz <user-id> [--debug] [--delete]
```

Options:

- `--debug` : print verbose debugging output to stdout
- `--delete`: delete the `.tar.gz` packet after successful processing

Example:

```bash
python ai.py latest_output_package.tar.gz 12345 --debug
```

This will process the images and audio inside `latest_output_package.tar.gz`, synthesize an event, print debugging information (if requested), and POST the event to the server.

## Where results are posted

The script currently posts to a hardcoded endpoint in `ai.py`:

```python
url = f"http://192.168.68.150:8080/api/add-event/{event_id}"
```

Make sure you either run an API server at that address that accepts the posted JSON, or modify `ai.py` to point to your API. The posted payload looks like:

```json
{
	"title": "...",
	"summary": ["bullet 1", "bullet 2"],
	"type": "Experience|Moment",
	"datetime": 169xxxxx,
	"actions": [{"title": "...", "start": 169xxx, "end": 169xxx}, ...]
}
```

## How it works (high-level)

1. Extract files from the provided `.tar.gz` packet.
2. Load and validate images; cluster them using GPT and pick representative images.
3. For each selected image, call the vision model to extract any actionable calendar information and short summary bullets.
4. Transcribe the audio using Whisper-like model, extract calendar data from the transcript.
5. Deduplicate similar events via GPT, synthesize a final short summary (4–5 bullets) and generate a concise event title.
6. Post the final event JSON to the configured HTTP API endpoint.

## Requirements

- Python 3.10+
- openai (or the repository's OpenAI client package as used in `ai.py`)
- pillow
- requests
- dateparser

Add these to `requirements.txt` if you want reproducible installs. A suggested `requirements.txt` is:

```
openai
Pillow
requests
dateparser

# pin versions as desired
```

## Development notes

- The script reads the API key from `API_KEY.txt`. If you'd prefer environment variables, change the `ai.py` load logic accordingly.
- The endpoint for posting events is hardcoded. Consider making it configurable via CLI flag or environment variable.
- Error handling: the code attempts to sanitize LLM outputs into JSON; however, LLMs may return non-JSON content. If you hit parsing errors, try `--debug` to see the raw outputs.

## Limitations & cost

- This project uses multiple LLM calls and image encoding; expect costs when using OpenAI models. Use conservative testing (e.g., single images) when iterating.
- Models may hallucinate or extract incorrect times/titles. Always review produced calendar events before trusting them.

## Example

If you'd like to test with the included sample packet folder `latest_output_package/`, first create a `.tar.gz` of that folder, or run:

```bash
tar -czf latest_output_package.tar.gz -C latest_output_package .
python ai.py latest_output_package.tar.gz 42 --debug
```

## TODO / Improvements

- Add CLI flags to configure the POST endpoint and API key source.
- Add tests for the parsing helpers and LLM output sanitization.
- Add unit tests and a small integration test that runs on a small sample packet.

## License

MIT

