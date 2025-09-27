#!/usr/bin/env python3
"""
AI Pipeline
- Unpack a tar.gz packet (PNG/JPG images + one WAV)
- Semantic clustering of images (gpt-4.1)
- For each cluster, GPT chooses the best image -> only that image is processed
- Process chosen images and audio -> extract events + summaries
- Apply strict deduplication (title exact + GPT similarity)
- Condense summary, generate title (gpt-4.1)
- Return Event object
"""

import os
import io
import re
import json
import tarfile
import tempfile
import shutil
import base64
import datetime
from typing import List, Tuple

import json
import requests
from typing import Union

import dateparser
from openai import OpenAI
from PIL import Image, UnidentifiedImageError

# ---------------------------
# Config
# ---------------------------

with open("API_KEY.txt", "r") as f:
    API_KEY = f.read()

if not API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=API_KEY)

# ---------------------------
# Helpers
# ---------------------------
def now():
    return datetime.datetime.now()

def parse_datetime_or_none(dt_str: str):
    """Try ISO parse or dateparser; return None on failure or if empty."""
    if not dt_str:
        return None
    try:
        return datetime.datetime.fromisoformat(dt_str)
    except Exception:
        parsed = dateparser.parse(dt_str)
        return parsed

def ensure_event_times(start_str: str, end_str: str) -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Convert start/end strings (possibly empty) to datetimes.
    - If start missing -> start = now()
    - If end missing -> end = start + 1 hour
    """
    st = parse_datetime_or_none(start_str)
    en = parse_datetime_or_none(end_str)

    if st is None:
        st = now()
    if en is None:
        en = st + datetime.timedelta(hours=1)
    # if en < st, set en = st + 1h
    if en <= st:
        en = st + datetime.timedelta(hours=1)
    return st, en

def clean_json_string(json_str: str) -> str:
    # Remove comments (// ...)
    s = re.sub(r"//.*?$", "", json_str, flags=re.MULTILINE)
    # Remove trailing commas (optional, in case JSON has them)
    s = re.sub(r",\s*]", "]", s)
    s = re.sub(r",\s*}", "}", s)
    return s.strip()

def safe_open_image_bytes(img_bytes: bytes):
    return Image.open(io.BytesIO(img_bytes))

# ---------------------------
# Data classes
# ---------------------------
class CalendarAction:
    def __init__(self, event: str, start: datetime.datetime, end: datetime.datetime):
        self.event = event
        self.datetime_start = start
        self.datetime_end = end

    def to_json(self):
        return {
            "title": self.event,
            "start": self.datetime_start.isoformat(),
            "end": self.datetime_end.isoformat()
        }

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

# ---------------------------
# Core processors
# ---------------------------
def process_image(image_bytes: bytes, debug: bool = False) -> Tuple[List[dict], List[str]]:
    """
    Return events as dicts (may have empty start/end) and summary list.
    We'll normalize times later with ensure_event_times.
    """
    try:
        safe_open_image_bytes(image_bytes)
    except UnidentifiedImageError as e:
        if debug:
            print(f"⚠️ process_image: invalid image, skipping: {e}")
        return [], []

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = client.responses.create(
        model="gpt-4.1",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text":
                    ("Analyze this image and ONLY extract actionable events, reminders, "
                     "deadlines, meetings, or decisions that are clearly implied. "
                     "Only extract events if the image clearly shows intentional information "
                     "(slides, whiteboards, notes, documents). If no actionable event is evident, "
                     "return empty 'events' and supply a short, useful summary (1-3 bullets). "
                     "DO NOT output filler like 'No actionable events'.\n\n"
                     "Return JSON: {\"events\": [{\"title\":\"...\",\"start\":\"ISO or empty\",\"end\":\"ISO or empty\"}], \"summary\": [\"...\"] }")
                },
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}
            ]
        }]
    )

    raw = resp.output_text.strip()
    if debug:
        print("\n📸 [DEBUG] process_image raw output:\n", raw)

    try:
        parsed = json.loads(clean_json_string(raw))
    except Exception:
        parsed = {"events": [], "summary": [raw]}

    events = parsed.get("events", []) if isinstance(parsed.get("events", []), list) else []
    summary = parsed.get("summary", []) if isinstance(parsed.get("summary", []), list) else []
    # filter trivial "no actionable events" texts
    summary = [s for s in summary if s and s.strip().lower() not in ("no actionable events", "no actionable events, reminders, etc.")]
    return events, summary

def process_audio(audio_bytes: bytes, debug: bool = False) -> Tuple[List[dict], List[str]]:
    """
    Transcribe with whisper-1, then extract events + summary via chat model.
    Returns events as dicts (start/end strings possibly empty) and summary list.
    """
    # Transcribe
    transcript_resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", io.BytesIO(audio_bytes))
    )
    transcript = transcript_resp.text
    if debug:
        print("\n🔊 [DEBUG] Transcript:\n", transcript)

    prompt = (
        "You are an assistant that reads meeting transcripts and extracts actionable events, "
        "deadlines, meeting times, and follow-ups. Ignore filler like 'No actionable events'.\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Return JSON: {\"events\": [{\"title\":\"...\",\"start\":\"ISO or empty\",\"end\":\"ISO or empty\"}], \"summary\": [\"...\"] }"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"You are a planner assistant."},
                  {"role":"user","content":prompt}],
        temperature=0.2
    )

    raw = response.choices[0].message.content.strip()
    if debug:
        print("\n📋 [DEBUG] process_audio raw output:\n", raw)

    try:
        parsed = json.loads(clean_json_string(raw))
    except Exception:
        parsed = {"events": [], "summary": [raw]}

    events = parsed.get("events", []) if isinstance(parsed.get("events", []), list) else []
    summary = parsed.get("summary", []) if isinstance(parsed.get("summary", []), list) else []
    summary = [s for s in summary if s and s.strip().lower() not in ("no actionable events", "no actionable events, reminders, etc.")]
    return events, summary

# ---------------------------
# Clustering & best-image selection
# ---------------------------
def describe_images_with_gpt(image_bytes_list: List[bytes], debug: bool = False) -> List[str]:
    descriptions = []
    for idx, img_bytes in enumerate(image_bytes_list):
        try:
            safe_open_image_bytes(img_bytes)
        except UnidentifiedImageError as e:
            if debug:
                print(f"⚠️ describe_images_with_gpt: unreadable image {idx}: {e}")
            descriptions.append("")
            continue

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        resp = client.responses.create(
            model="gpt-4.1",
            input=[{
                "role": "user",
                "content": [
                    {"type":"input_text","text":"Describe the key, relevant content of this image in 1-2 concise sentences. Mention objects, text, or signs it was intentionally captured (whiteboard/slide/document)."},
                    {"type":"input_image","image_url":f"data:image/png;base64,{b64}"}
                ]
            }]
        )
        desc = resp.output_text.strip()
        if debug:
            print(f"\n🖼️ [DEBUG] image {idx} description:\n", desc)
        descriptions.append(desc)
    return descriptions

def cluster_images_via_gpt(descriptions: List[str], debug: bool = False) -> List[List[int]]:
    enumerated = "\n".join(f"{i}: {desc if desc else '[unreadable]'}" for i, desc in enumerate(descriptions))
    prompt = (
        "Group the numbered image descriptions below into clusters of visually/semantically similar images. "
        "Return a JSON array of clusters (each cluster is a list of indices). Every index must appear exactly once. "
        "Group unreadable images separately.\n\n"
        f"{enumerated}"
    )
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role":"user","content":prompt}],
        temperature=0.01,
        max_tokens=800
    )
    raw = resp.choices[0].message.content.strip()
    if debug:
        print("\n🗂️ [DEBUG] GPT clustering raw output:\n", raw)
    try:
        clusters = json.loads(clean_json_string(raw))
        if not isinstance(clusters, list) or not all(isinstance(c, list) for c in clusters):
            raise ValueError("Malformed clusters")
    except Exception:
        clusters = [[i for i, d in enumerate(descriptions) if d != ""]]
    return clusters

def choose_best_image_in_cluster(cluster_idx_list: List[int], descriptions: List[str], images: List[bytes], debug: bool = False) -> int:
    """
    For the given indices, ask GPT (gpt-4.1) to pick the single most informative image.
    We provide index, description, and a small hint that we want clarity and intentional capture.
    Return chosen index (absolute).
    """
    candidates = []
    for idx in cluster_idx_list:
        candidates.append({"index": idx, "description": descriptions[idx] if idx < len(descriptions) else ""})
    payload = json.dumps(candidates, indent=2)
    prompt = (
        "You are an assistant that, given a small set of similar images, must pick the single most informative photo.\n"
        "Each item has 'index' and 'description'. Choose the image index that best represents the cluster (most clear, intentionally captured content). "
        "Return EXACT JSON: {\"chosen_index\": <integer>}.\n\n"
        f"Candidates:\n{payload}"
    )
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role":"user","content":prompt}],
        temperature=0.01,
        max_tokens=200
    )
    raw = resp.choices[0].message.content.strip()
    if debug:
        print("\n🔬 [DEBUG] choose_best_image raw output:\n", raw)
    try:
        picked = json.loads(clean_json_string(raw))
        chosen = picked.get("chosen_index")
        if chosen not in cluster_idx_list:
            raise ValueError("chosen not in cluster")
    except Exception:
        # fallback heuristic: pick description with most words
        best = cluster_idx_list[0]
        best_len = -1
        for idx in cluster_idx_list:
            desc = descriptions[idx] if idx < len(descriptions) else ""
            L = len(desc.split())
            if L > best_len:
                best_len = L
                best = idx
        chosen = best
        if debug:
            print("⚠️ choose_best_image fallback to heuristic:", chosen)
    if debug:
        print("✅ chosen image index for cluster:", chosen)
    return chosen

# ---------------------------
# Deduplication (strict, unchanged)
# ---------------------------
def deduplicate_events_strict(events: List[CalendarAction], debug: bool = False) -> List[CalendarAction]:
    if not events:
        return []
    # Stage 1: exact-title dedupe
    unique_by_title = {}
    for e in events:
        key = e.event.lower().strip()
        if key not in unique_by_title:
            unique_by_title[key] = e
    events_stage1 = list(unique_by_title.values())

    # Stage 2: GPT similarity dedupe (strict)
    events_json = json.dumps([ev.to_json() for ev in events_stage1], indent=2)
    prompt = (
        "You are an expert calendar assistant. STRICTLY remove any events that are similar in title to another event, "
        "even if timestamps differ. Do NOT merge; only keep distinct titles. Return a cleaned JSON list of events."
        f"\n\nEvents:\n{events_json}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"You are an expert calendar assistant."},
                  {"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=1000
    )
    raw = resp.choices[0].message.content.strip()
    if debug:
        print("\n🗂️ [DEBUG] deduplicate_events_strict raw output:\n", raw)
    try:
        deduped = json.loads(clean_json_string(raw))
    except Exception:
        deduped = [ev.to_json() for ev in events_stage1]

    cleaned = []
    for ev in deduped:
        try:
            start_str = ev.get("start", "")
            end_str = ev.get("end", "")
            st, en = ensure_event_times(start_str, end_str)
            cleaned.append(CalendarAction(ev.get("title","").strip(), st, en))
        except Exception:
            continue
    return cleaned

# ---------------------------
# Full pipeline
# ---------------------------
def process_event(images: List[bytes], audio: bytes, debug: bool = False) -> Event:
    # 1) Describe images
    descriptions = describe_images_with_gpt(images, debug=debug)

    # 2) Cluster via GPT
    clusters_idx = cluster_images_via_gpt(descriptions, debug=debug)
    if debug:
        print("\n🔎 [DEBUG] clusters:", clusters_idx)

    # 3) For each cluster: GPT chooses best image; process that one only
    cluster_level_events = []
    cluster_level_summaries = []
    for cluster in clusters_idx:
        # sanitize indices
        cluster_valid = [i for i in cluster if 0 <= i < len(images)]
        if not cluster_valid:
            continue
        chosen_idx = choose_best_image_in_cluster(cluster_valid, descriptions, images, debug=debug)
        # Process chosen image
        events_dicts, summary = process_image(images[chosen_idx], debug=debug)
        # Normalize times and convert to CalendarAction
        for ev in events_dicts:
            title = ev.get("title","").strip()
            st_str = ev.get("start","")
            end_str = ev.get("end","")
            st, en = ensure_event_times(st_str, end_str)
            cluster_level_events.append(CalendarAction(title, st, en))
        cluster_level_summaries.extend(summary)

    # 4) Process audio
    audio_events_dicts, audio_summary = process_audio(audio, debug=debug)
    for ev in audio_events_dicts:
        title = ev.get("title","").strip()
        st_str = ev.get("start","")
        end_str = ev.get("end","")
        st, en = ensure_event_times(st_str, end_str)
        cluster_level_events.append(CalendarAction(title, st, en))
    cluster_level_summaries.extend(audio_summary)

    # 5) Strict dedupe
    if debug:
        print("\n🗂️ [DEBUG] Running strict deduplication on combined events...")
    final_actions = deduplicate_events_strict(cluster_level_events, debug=debug)

    # 6) Summaries: dedupe locally and condense to 3-4 bullets with GPT
    unique_summary = list(dict.fromkeys([s.strip() for s in cluster_level_summaries if s and s.strip()]))
    summary_text = "\n".join(unique_summary) if unique_summary else ""
    condensed_summary = []
    if summary_text:
        condense_prompt = (
            "Condense the following bullet points into the 3-4 most important, non-redundant bullet points. "
            "Return EXACT JSON array of strings.\n\n" + summary_text
        )
        cond_resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role":"user","content":condense_prompt}],
            temperature=0.0,
            max_tokens=400
        )
        raw_cond = cond_resp.choices[0].message.content.strip()
        if debug:
            print("\n🧠 [DEBUG] condense raw output:\n", raw_cond)
        try:
            condensed_summary = json.loads(clean_json_string(raw_cond))
            if not isinstance(condensed_summary, list):
                condensed_summary = unique_summary[:4]
        except Exception:
            condensed_summary = unique_summary[:4]
    else:
        condensed_summary = []

    # 7) Title generation (gpt-4.1)
    title = "Untitled Event"
    if condensed_summary:
        title_prompt = (
            "Generate a concise, informative title (6 words max) for an event based on the following summary bullets. "
            "Return only the title string.\n\n" + "\n".join(condensed_summary)
        )
        t_resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role":"user","content":title_prompt}],
            temperature=0.2,
            max_tokens=30
        )
        title_raw = t_resp.choices[0].message.content.strip()
        if debug:
            print("\n🏷️ [DEBUG] title raw output:\n", title_raw)
        title = title_raw.splitlines()[0].strip() if title_raw else title

    return Event(title=title, summary=condensed_summary, actions=final_actions, moment=(len(images) <= 15))

# ---------------------------
# Tar.gz wrapper
# ---------------------------
def process_packet_tar_gz(tar_path: str, debug: bool = False) -> Event:
    temp_dir = tempfile.mkdtemp()
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    images = []
    audio = None
    for root, _, files in os.walk(temp_dir):
        for f in files:
            path = os.path.join(root, f)
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    with open(path, "rb") as fh:
                        images.append(fh.read())
                except Exception:
                    continue
            elif f.lower().endswith(".wav"):
                with open(path, "rb") as fh:
                    audio = fh.read()

    if not images:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError("No images found in the tar.gz packet.")
    if audio is None:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError("No WAV audio file found in the tar.gz packet.")

    try:
        evt = process_event(images, audio, debug=debug)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return evt

def post_event(event: "Event", event_id: Union[str, int], debug: bool = False) -> bool:
    """
    POST an Event object as JSON to localhost/api/add-event/{id}

    Args:
        event: Event object to send
        event_id: ID for the endpoint URL
        debug: If True, prints payload and response

    Returns:
        True if status code 200-299, False otherwise
    """
    url = f"http://localhost/api/add-event/{event_id}"
    payload = {
        "title": event.title,
        "summary": event.summary,
        "type": event.type,
        "datetime": event.datetime.isoformat(),
        "actions": [
            {
                "title": a.event,
                "start": a.datetime_start.isoformat(),
                "end": a.datetime_end.isoformat()
            } for a in event.actions
        ]
    }

    if debug:
        print("🚀 Posting Event to API:", url)
        print(json.dumps(payload, indent=2))

    try:
        resp = requests.post(url, json=payload)
        if debug:
            print("📌 Response:", resp.status_code, resp.text)
        return 200 <= resp.status_code < 300
    except Exception as e:
        if debug:
            print("❌ Error posting event:", e)
        return False


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process packet tar.gz into an Event.")
    parser.add_argument("tarfile", help="Path to the .tar.gz packet")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    final_event = process_packet_tar_gz(args.tarfile, debug=args.debug)
    print("\n✅ Final Event:\n", final_event)

    post_event(final_event, "68d815d73a56a6fa6fccdf24")