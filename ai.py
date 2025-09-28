import datetime
import json
import io
import os
import re
import tarfile
import tempfile
import shutil
import base64
from typing import List, Tuple, Union
import requests
from PIL import Image, UnidentifiedImageError
from openai import OpenAI
import dateparser

# --- Load API Key ---
with open("API_KEY.txt", "r") as f:
    API_KEY = f.read().strip()
client = OpenAI(api_key=API_KEY)

# --- Helpers ---
def parse_datetime_with_current_year(dt_str: str) -> datetime.datetime:
    now = datetime.datetime.now()
    if not dt_str:
        return now
    try:
        dt = datetime.datetime.fromisoformat(dt_str)
    except ValueError:
        dt = dateparser.parse(dt_str) or now

    if dt.year < now.year:
        dt = dt.replace(year=now.year)
    return dt

def clean_json_string(json_str: str) -> str:
    # Remove backticks, comments, trailing commas
    s = re.sub(r"^```(?:json)?\s*", "", json_str)
    s = re.sub(r"```$", "", s)
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r",\s*(\]|\})", r"\1", s)
    return s.strip()

def filter_valid_images(images: List[bytes]) -> List[bytes]:
    valid_images = []
    for img_bytes in images:
        try:
            Image.open(io.BytesIO(img_bytes))
            valid_images.append(img_bytes)
        except UnidentifiedImageError:
            continue
    return valid_images

# --- Core Classes ---
class CalendarAction:
    def __init__(self, event: str, start: datetime.datetime, end: datetime.datetime):
        self.event = event
        self.datetime_start = start
        self.datetime_end = end

    def __repr__(self):
        return f"<CalendarAction {self.event} ({self.datetime_start} - {self.datetime_end})>"
    
    def __str__(self):
        return f"Calendar event for {self.event}"

class Event:
    def __init__(self, title: str, summary: List[str], actions: List[CalendarAction], moment: bool):
        self.title = title
        self.summary = summary
        self.actions = actions
        self.type = "Moment" if moment else "Experience"
        self.datetime = datetime.datetime.now()

    def __repr__(self):
        return f"<Event title='{self.title}' summary={self.summary} actions={len(self.actions)}>"

# --- Image Clustering & Best Selection ---
# --- GPT-Based Image Clustering ---
def cluster_images(images: List[bytes], debug: bool = False) -> List[bytes]:
    """
    Use GPT to cluster images based on content similarity.
    Returns the best image per cluster.
    """
    descriptions = []
    for i, img_bytes in enumerate(images):
        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        prompt_text = (
            "Describe this image in 1-2 short sentences, focusing on important info "
            "that could be relevant for events or reminders. Be concise."
        )
        response = client.responses.create(
            model="gpt-4.1",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"}
                ]
            }]
        )
        desc = response.output_text.strip()
        descriptions.append(desc)
        if debug:
            print(f"Image {i} description:", desc)

    # Now cluster descriptions using GPT
    cluster_prompt = (
        "Given the following image descriptions, group them into clusters where images "
        "seem related or show very similar content (they should be almost precisely exact in similarity of content). It is ok to not create a cluster if you have a small amount of image descriptions and they're all different - just make each one their own cluster. Return a JSON list of clusters, each containing "
        "the indices of images in that cluster. Do not output anything else apart from the JSON. Example: [{\"cluster\": [0, 1, 2]}, {\"cluster\": [3, 4]}, ...]\n"
        f"Descriptions:\n{json.dumps(descriptions)}"
    )

    cluster_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": cluster_prompt}],
        temperature=0.01
    )

    raw_clusters = cluster_resp.choices[0].message.content.strip()
    if debug:
        print("GPT raw clusters:", clean_json_string(raw_clusters))

    cleaned = clean_json_string(raw_clusters)
    try:
        clusters = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: each image is its own cluster
        clusters = [[i] for i in range(len(images))]

    # Select the "best" image per cluster (most content)
    best_images = []
    for cluster in clusters:
        max_content = -1
        best_img = None
        for idx in cluster["cluster"]:
            try:
                img = Image.open(io.BytesIO(images[idx])).convert("L")
                content = sum(1 for px in img.getdata() if px < 250)
                if content > max_content:
                    max_content = content
                    best_img = images[idx]
            except UnidentifiedImageError:
                continue
        if best_img:
            best_images.append(best_img)

    if debug:
        print(f"GPT clustered {len(images)} images into {len(best_images)} best images.")

    return best_images

# --- Image Processing ---
def process_image(image_bytes: bytes, debug: bool = False) -> Tuple[List[CalendarAction], List[str]]:
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    prompt_text = (
        "Strictly return valid JSON. Extract **all information necessary for a calendar reminder**: "
        "event title, start and end times."
        "Include at most one actionable event per image. If no event, return an empty 'events' list. "
        "Also provide 1-3 concise summary bullets of the important points discussed - any descriptions of data, images, or relevant content in the foreground of the picture. Don't feel any need to describe the scenery or setting or generate useless content, if there isn't anything valuable to describe, return an empty list under the summary section.\n\n"
        "Return JSON like this:\n"
        "{'events':[{'title':..., 'start':..., 'end':...}, ...], 'summary':[...]}."
    )

    response = client.responses.create(
        model="gpt-4.1",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"}
            ]
        }]
    )

    raw_output = response.output_text.strip()
    if debug:
        print("Image raw output:", raw_output)

    cleaned = clean_json_string(raw_output)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = {"events": [], "summary": [cleaned]}

    events = []
    for e in parsed.get("events", []):
        start = parse_datetime_with_current_year(e.get("start"))
        end = parse_datetime_with_current_year(e.get("end")) if e.get("end") else start + datetime.timedelta(hours=1)
        events.append(CalendarAction(
            event=e.get("title", "Untitled Event"),
            start=start,
            end=end
        ))

    return events, parsed.get("summary", [])

# --- Audio Processing ---
def process_audio(audio_bytes: bytes, debug: bool = False) -> Tuple[List[CalendarAction], List[str]]:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("audio.wav", io.BytesIO(audio_bytes))
    ).text

    prompt_text = (
        "You are a planning assistant. Extract **all information needed to create calendar reminders** from this transcript: "
        "event title, start and end times."
        "Include at most one actionable event per transcript. If no event is found, return an empty 'events' list. "
        "Also provide 1-3 concise summary bullets of the important points discussed - any descriptions of data, images, or relevant content in the foreground of the picture. Don't feel any need to describe the scenery or setting or generate useless content, if there isn't anything valuable to describe, return an empty list under the summary section\n\n"
        "Transcript:\n" + transcript + "\n\n"
        "Return strictly valid JSON like this:\n"
        "{'events':[{'title':..., 'start':..., 'end':...], 'summary':[...]}."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a planner assistant."},
                  {"role": "user", "content": prompt_text}],
        temperature=0.0
    )

    raw_output = response.choices[0].message.content.strip()
    if debug:
        print("Audio raw output:", raw_output)

    cleaned = clean_json_string(raw_output)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = {"events": [], "summary": [cleaned]}

    events = []
    for e in parsed.get("events", []):
        start = parse_datetime_with_current_year(e.get("start"))
        end = parse_datetime_with_current_year(e.get("end")) if e.get("end") else start + datetime.timedelta(hours=1)
        events.append(CalendarAction(
            event=e.get("title", "Untitled Event"),
            start=start,
            end=end
        ))

    return events, parsed.get("summary", [])

# --- Deduplication ---
def deduplicate_events(events: List[CalendarAction], debug: bool = False) -> List[CalendarAction]:
    events_json = [{"title": e.event, "start": e.datetime_start.isoformat(), "end": e.datetime_end.isoformat()} for e in events]
    prompt = (
        "Deduplicate events by title similarity (ignore times). "
        "Return only the JSON list of distinct events. Do not return any other dialogue except for the JSON. \n" + json.dumps(events_json)
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    raw = response.choices[0].message.content.strip()
    if debug:
        print("Deduplication raw output:", raw)

    cleaned = clean_json_string(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return events

    deduped = []
    for e in parsed:
        start = parse_datetime_with_current_year(e.get("start"))
        end = parse_datetime_with_current_year(e.get("end")) if e.get("end") else start + datetime.timedelta(hours=1)
        deduped.append(CalendarAction(e.get("title", "Untitled Event"), start, end))
    return deduped

# --- Event Compilation ---
def process_event(images: List[bytes], audio: bytes, debug: bool = False) -> Event:
    images = filter_valid_images(images)
    images = cluster_images(images, debug=debug)

    all_actions = []
    all_summaries = []

    for img in images:
        acts, summ = process_image(img, debug=debug)
        all_actions.extend(acts)
        all_summaries.extend(summ)

    audio_actions, audio_summary = process_audio(audio, debug=debug)
    all_actions.extend(audio_actions)
    all_summaries.extend(audio_summary)

    all_actions = deduplicate_events(all_actions, debug=debug)
    
    # Deduplicate summary bullets
    unique_summary = list({s.strip(): None for s in all_summaries}.keys())

    # Use ChatGPT to synthesize into 4-5 key bullets
    summary_prompt = (
        "You are an assistant that condenses a list of event summary bullets into 4-5 "
        "concise, clear, and important bullet points. Make sure to save as many distinct details as possible - all the meaning from the original bullets should be properly encapsulated in the final summary. Original bullets:\n"
        + "\n".join(unique_summary)
    )
    summary_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.4
    )
    synthesized_summary = summary_resp.choices[0].message.content.strip().split("\n")
    # Keep only non-empty bullets and limit to 5
    synthesized_summary = [s.strip("-• ").strip() for s in synthesized_summary if s.strip()][:5]

    # Generate title from both actions and the synthesized summary
    title_prompt = (
        "Generate a concise, engaging event title based on both the following actions "
        "and summary bullets. The title should capture the essence of the event, be short and clear.\n\n"
        "Actions:\n" + "\n".join([str(action) for action in all_actions]) + "\n\n"
        "Summary Bullets:\n" + "\n".join(synthesized_summary)
    )
    title_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": title_prompt}],
        temperature=0.4
    )
    title = title_resp.choices[0].message.content.strip()

    return Event(title=title, summary=synthesized_summary, actions=all_actions, moment=(len(images) <= 15))

# --- Tar.gz Processing ---
def process_packet_tar_gz(tar_path: str, debug: bool = False, delete_at_end: bool = False) -> Event:
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=temp_dir)

    images, audio = [], None
    for root, _, files in os.walk(temp_dir):
        for f in files:
            fp = os.path.join(root, f)
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                with open(fp, "rb") as img_file:
                    images.append(img_file.read())
            elif f.lower().endswith(".wav"):
                with open(fp, "rb") as audio_file:
                    audio = audio_file.read()
    shutil.rmtree(temp_dir, ignore_errors=True)

    if not images:
        raise ValueError("No valid images found in the tar.gz packet.")
    if not audio:
        raise ValueError("No WAV audio file found in the tar.gz packet.")
    
    if delete_at_end:
        os.remove(tar_path)

    return process_event(images, audio, debug=debug)

# --- Post Event ---
def post_event(event: Event, event_id: Union[str, int], debug: bool = False) -> bool:
    url = f"http://192.168.68.150:8080/api/add-event/{event_id}"
    payload = {
        "title": event.title,
        "summary": event.summary,
        "type": event.type,
        "datetime": int(event.datetime.timestamp()),
        "actions": [{"title": a.event, "start": int(a.datetime_start.timestamp()),
                     "end": int(a.datetime_end.timestamp())} for a in event.actions]
    }
    if debug:
        print("Posting Event JSON:", json.dumps(payload, indent=2))
    try:
        resp = requests.post(url, json=payload)
        if debug:
            print("API Response:", resp.status_code, resp.text)
        return 200 <= resp.status_code < 300
    except Exception as e:
        if debug:
            print("Error posting event:", e)
        return False

# --- Example ---
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Process packet tar.gz into an Event.")
    parser.add_argument("tarfile", help="Path to the .tar.gz packet")
    parser.add_argument("id", help="User ID")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--delete", action="store_true", help="Delete file at the end")
    args = parser.parse_args()

    final_event = process_packet_tar_gz(args.tarfile, debug=args.debug, delete_at_end=args.delete)
    print("\n✅ Final Event:\n", final_event)

    post_event(final_event, args.id, debug=args.debug)
