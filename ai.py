import datetime
import json
from typing import List, Tuple
from openai import OpenAI
import dateparser
import tarfile
import tempfile
import os
import io
import shutil
import base64
import re

with open("API_KEY.txt") as f:
    API_KEY = f.read().strip()

client = OpenAI(api_key=API_KEY)


# --- Helpers ---
def parse_datetime(dt_str: str) -> datetime.datetime:
    try:
        return datetime.datetime.fromisoformat(dt_str)
    except ValueError:
        parsed = dateparser.parse(dt_str)
        return parsed if parsed else datetime.datetime.now()


class CalendarAction:
    def __init__(self, event: str, start: datetime.datetime, end: datetime.datetime):
        self.event = event
        self.datetime_start = start
        self.datetime_end = end

    def __repr__(self):
        return f"<CalendarAction {self.event} ({self.datetime_start} - {self.datetime_end})>"


class Event:
    def __init__(self, title: str, summary: List[str], actions: List[CalendarAction], moment: bool):
        self.title = title
        self.summary = summary
        self.actions = actions
        self.type = "Moment" if moment else "Experience"
        self.datetime = datetime.datetime.now()

    def __repr__(self):
        return f"<Event title='{self.title}' summary={self.summary} actions={len(self.actions)}>"


def clean_json_string(json_str: str) -> str:
    """
    Remove wrapping backticks, ```json or ``` code blocks from a string.
    """
    # Remove ```json or ``` at the start
    json_str = re.sub(r"^```(?:json)?\s*", "", json_str.strip())
    # Remove ``` at the end
    json_str = re.sub(r"```$", "", json_str.strip())
    return json_str

# ---------- Process Image ----------
def process_image(image_bytes: bytes, debug: bool = False) -> Tuple[List[CalendarAction], List[str]]:
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Analyze this image and ONLY extract actionable events, reminders, "
                            "deadlines, meetings, or decisions that are clearly implied.\n"
                            "- Only consider images where the camera is intentionally capturing "
                            "important information (e.g., whiteboards, slides, notes, documents).\n"
                            "- If the image does not clearly contain intentional, valuable info, return an empty events list.\n"
                            "- Do not include summaries like 'No actionable events, reminders, etc.'\n"
                            "- Return JSON with:\n"
                            "  - 'events': list of {title, start, end}\n"
                            "  - 'summary': list of 1–3 meaningful key points."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    raw_output = response.output_text.strip()
    if debug:
        print("\n📸 [DEBUG] Raw image output:\n", raw_output, "\n")

    try:
        parsed = json.loads(clean_json_string(raw_output))
    except json.JSONDecodeError:
        parsed = {"events": [], "summary": [raw_output]}

    events = []
    for e in parsed.get("events", []):
        try:
            events.append(
                CalendarAction(
                    event=e["title"],
                    start=parse_datetime(e["start"]),
                    end=parse_datetime(e["end"]),
                )
            )
        except Exception:
            continue

    # Filter out meaningless summaries
    summary = [
        s for s in parsed.get("summary", [])
        if s.strip().lower() not in ["no actionable events", "no actionable events, reminders, etc."]
    ]

    return events, summary


# ---------- Process Audio ----------
def process_audio(audio_bytes: bytes, debug: bool = False) -> Tuple[List[CalendarAction], List[str]]:
    # 1️⃣ Transcribe
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", io.BytesIO(audio_bytes))
    ).text

    if debug:
        print("\n🔊 [DEBUG] Transcript:\n", transcript, "\n")

    # 2️⃣ Summarize & extract events
    prompt = (
        "You are an assistant that reads transcripts and extracts useful reminders, events, "
        "meetings, and deadlines.\n"
        "- Do not include summaries that say 'No actionable events' or similar.\n"
        "- Focus only on meaningful actionable items.\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Return JSON with:\n"
        "- 'events': list of {title, start, end}\n"
        "- 'summary': list of 1–3 key actionable takeaways."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a planner assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    raw_output = response.choices[0].message.content.strip()
    if debug:
        print("\n📋 [DEBUG] Audio analysis output:\n", raw_output, "\n")

    try:
        parsed = json.loads(clean_json_string(raw_output))
    except json.JSONDecodeError:
        parsed = {"events": [], "summary": [raw_output]}

    events = []
    for e in parsed.get("events", []):
        try:
            events.append(
                CalendarAction(
                    event=e["title"],
                    start=parse_datetime(e["start"]),
                    end=parse_datetime(e["end"]),
                )
            )
        except Exception:
            pass

    summary = [
        s for s in parsed.get("summary", [])
        if s.strip().lower() not in ["no actionable events", "no actionable events, reminders, etc."]
    ]

    return events, summary


# ---------- Final Event Processing ----------
def process_event(images: List[bytes], audio: bytes, debug: bool = False) -> Event:
    all_summaries = []
    all_actions = []

    # Process all images
    for i, img in enumerate(images, 1):
        if debug:
            print(f"\n🔎 [DEBUG] Processing image {i}/{len(images)}...")
        actions, summary = process_image(img, debug=debug)
        if actions or summary:  # Skip images with no meaningful info
            all_actions.extend(actions)
            all_summaries.extend(summary)

    # Process audio
    if debug:
        print("\n🔎 [DEBUG] Processing audio...")
    audio_actions, audio_summary = process_audio(audio, debug=debug)
    all_actions.extend(audio_actions)
    all_summaries.extend(audio_summary)

    # Deduplicate actions
    unique_actions = []
    seen = set()
    for action in all_actions:
        key = (action.event.lower(), action.datetime_start)
        if key not in seen:
            unique_actions.append(action)
            seen.add(key)

    # Deduplicate summaries
    unique_summary = list({s.strip(): None for s in all_summaries}.keys())

    # Condense summary to 3-4 points
    summary_text = "\n".join(unique_summary)
    condensation_prompt = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Condense the following bullet points into 4-5 of the most important points. Return only a JSON list of strings."},
            {"role": "user", "content": summary_text},
        ],
        max_tokens=300,
        temperature=0.3
    )

    condensed_output = condensation_prompt.choices[0].message.content
    if debug:
        print("\n🧠 [DEBUG] Condensed summary raw output:\n", condensed_output, "\n")

    try:
        condensed_summary = json.loads(condensed_output)
    except json.JSONDecodeError:
        condensed_summary = unique_summary[:4]

    # Generate title
    title_prompt = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Generate a concise and informative title for an event based on the following summary."},
            {"role": "user", "content": "\n".join(condensed_summary)},
        ],
        max_tokens=50,
        temperature=0.4
    )
    title = title_prompt.choices[0].message.content.strip()
    if debug:
        print("\n🏷️ [DEBUG] Generated title:\n", title, "\n")

    return Event(
        title=title,
        summary=condensed_summary,
        actions=unique_actions,
        moment=(len(images) <= 15)
    )


# ---------- Process tar.gz ----------
def process_packet_tar_gz(tar_path: str, debug: bool = False) -> Event:
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=temp_dir)

    images = []
    audio = None

    for root, _, files in os.walk(temp_dir):
        for f in files:
            file_path = os.path.join(root, f)
            if f.lower().endswith((".jpg", ".png")):
                with open(file_path, "rb") as img_file:
                    images.append(img_file.read())
            elif f.lower().endswith(".wav"):
                with open(file_path, "rb") as audio_file:
                    audio = audio_file.read()

    if not images:
        raise ValueError("No images found in the tar.gz packet.")
    if audio is None:
        raise ValueError("No WAV audio file found in the tar.gz packet.")

    event = process_event(images, audio, debug=debug)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return event


# ---------- Main ----------
if __name__ == "__main__":
    # Enable debug for verbose output
    final_event = process_packet_tar_gz("output_package.tar.gz", debug=True)
    print("\n✅ Final Event:\n", final_event)
