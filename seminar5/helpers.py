import threading
import cv2
import numpy as np
import base64
import io
from PIL import Image
from openai import OpenAI
from typing import List
import re


class RTSPStreamReader:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not self.running: break
            with self.lock:
                self.ret = ret
                self.frame = frame
            if not ret: break

    def read(self):
        with self.lock:
            if self.frame is None: return self.ret, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()
        
def is_green_screen(frame):
    if frame is None: return False
    small = cv2.resize(frame, (64, 64))
    avg_color = np.mean(small, axis=(0, 1))
    return avg_color[1] > 200 and avg_color[0] < 50 and avg_color[2] < 50

def measure_global_change(f1, f2):
    # Resize to 350x250 (perfectly divisible by 7x5 grid)
    # This resolution is high enough to detect objects but small enough to fit in cache.
    w, h = 210, 150
    g1 = cv2.cvtColor(cv2.resize(f1, (w, h)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(f2, (w, h)), cv2.COLOR_BGR2GRAY)
    
    # Compute difference once for the whole image
    diff_map = cv2.absdiff(g1, g2)
    
    uw, uh = w // 7, h // 5  # Unit width (50px) and height (50px)
    maes = []

    # Sample 6 areas at (row, col) indices: (0,0), (0,3), (0,6), (4,0), (4,3), (4,6)
    for r in [0, 4]:
        for c in [0, 3, 6]:
            roi_diff = diff_map[r*uh : (r+1)*uh, c*uw : (c+1)*uw]
            maes.append(np.mean(roi_diff))

    # Return minimum: Only trigger if ALL regions show significant change (Global Motion)
    return min(maes)


def get_formatted_date(date_string):
    """Your provided post-processing logic."""
    if not isinstance(date_string, str):
        return None
    date_string = re.sub(r'[^a-zA-Z0-9]', '', date_string)
    letter_to_number_map = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'R': '8', 'A': '4'} # Added 'A' for '4'
    date_string = ''.join(letter_to_number_map.get(char, char) for char in date_string)

    if len(date_string) == 8 and date_string.isdigit():
        #can add extra fixing rules per digit, e.g. first digit of month can only be 0 or 1, maybe we want to map all possible values to them
        if date_string[2] not in ["0","1"]:
            others_to_m1 = {'2': '1', '3': '0', '4': '1', '5': '0', '6': '0', '7': '1', '8': '0', '9': '0'} # 
            date_string = date_string[:2] + others_to_m1[date_string[2]] + date_string[3:]        
        
        day, month, year = int(date_string[:2]), int(date_string[2:4]), int(date_string[4:])
        datestr = f"{day:02}.{month:02}.{year:04}"
        if 1 <= day <= 31 and 1 <= month <= 12:
            return datestr
        else:
            #print("text not in Date range" + f"{day:02}.{month:02}.{year:04}")
            return datestr
    #else: 
    #    print("we had non-numbers in text:", date_string)
    return "NA"

def print_date_grid(detected_dates_ordered: List[str], expected_date_str: str):
    """
    Prints a 2x2 grid of detected dates with color coding.
    Assumes detected_dates_ordered is in the order: Top-Left, Bottom-Left, Top-Right, Bottom-Right.
    Colors: Green (match), Red (mismatch), Yellow (no date).
    """
    COLOR_RESET = "\033[0m"
    COLOR_GREEN = "\033[92m"
    COLOR_RED = "\033[91m"
    COLOR_YELLOW = "\033[93m"
    processed_dates = []
    for detected_raw in detected_dates_ordered:
        formatted_date = get_formatted_date(detected_raw)
        
        # Standardize missing dates to "N/A"
        display_text = formatted_date if formatted_date else "N/A"

        # Determine the color explicitly
        if display_text in ["N/A", "NA"]:
            color = COLOR_YELLOW
        elif display_text == expected_date_str:
            color = COLOR_GREEN
        else:
            color = COLOR_RED
            
        # Apply the 15-character padding to the raw text FIRST
        padded_text = f"{display_text:<15}"
        # Then wrap the perfectly sized text in color codes
        processed_dates.append(f"{color}{padded_text}{COLOR_RESET}")

    while len(processed_dates) < 4:
        # Pad the N/A string as well
        padded_na = f"{'N/A':<15}"
        processed_dates.append(f"{COLOR_YELLOW}{padded_na}{COLOR_RESET}")

    tl_date = processed_dates[0]
    bl_date = processed_dates[1]
    tr_date = processed_dates[2]
    br_date = processed_dates[3]

    print("\n--- Kuupäeva tuvastamise tulemused (2x2 ruudustik) ---")
    print(f"Oodatav aegumiskuupäev: {expected_date_str}")
    print(f"┌─────────────────┬─────────────────┐")
    # THE FIX PART 2: Remove the :<15 from here, because the spacing is already baked in!
    print(f"│ {tl_date} │ {tr_date} │")
    print(f"├─────────────────┼─────────────────┤")
    print(f"│ {bl_date} │ {br_date} │")
    print(f"└─────────────────┴─────────────────┘")


class OpenRouterOCR:
    """
    Klass OpenRouteri VLM-i (Vision Language Model) kasutamiseks OCR-i jaoks.
    Kapseldab API kliendi, viibad ja kasutusstatistika jälgimise.
    """
    def __init__(self, api_key: str, model: str, system_prompt: str, user_prompt: str, app_referer: str, app_title: str):
        """
        Initsialiseerib OpenRouterOCR klassi.

        Args:
            api_key (str): OpenRouter API võti.
            model (str): Kasutatava VLM-i mudeli nimi.
            system_prompt (str): Süsteemi viip OCR-i jaoks.
            user_prompt (str): Kasutaja viip OCR-i jaoks.
            app_referer (str): Rakenduse HTTP-Referer päis.
            app_title (str): Rakenduse X-Title päis.
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": app_referer,
                "X-Title": app_title,
            },
        )
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.usage_totals = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "last_call_info": None,
        }

    def _usage_to_dict(self, usage_obj) -> dict:
        """Teisendab OpenAI/OpenRouteri kasutusobjekti (CompletionUsage) sõnastikuks."""
        if usage_obj is None:
            return {}
        return {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
            "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
            "total_tokens": getattr(usage_obj, "total_tokens", 0),
            "cost": getattr(usage_obj, "cost", 0.0), # OpenRouter lisab 'cost'
        }

    def record_usage(self, usage_obj, *, step: str):
        """Kogub kokku kasutusstatistikad globaalsesse usage_totals sõnastikku."""
        u = self._usage_to_dict(usage_obj)
        if not u:
            return

        self.usage_totals["calls"] += 1
        self.usage_totals["prompt_tokens"] += int(u.get("prompt_tokens", 0))
        self.usage_totals["completion_tokens"] += int(u.get("completion_tokens", 0))
        self.usage_totals["total_tokens"] += int(u.get("total_tokens", 0))
        self.usage_totals["cost"] += float(u.get("cost", 0.0))
        self.usage_totals["last_call_info"] = {"step": step, **u}

    def tuvastus_openrouter(self, image_batch: List[np.ndarray]) -> List[str]:
        """
        Teostab OCR-i OpenRouteri VLM-i abil piltide partiil.
        Teeb iga pildi kohta eraldi API päringu.
        Tagastab toortekstide loendi, ühe iga pildi kohta partiis.
        """
        if not image_batch:
            return []

        raw_texts = []
        for img_rgb in image_batch:
            pil_img = Image.fromarray(img_rgb)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": [{"type": "text", "text": self.user_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}]}]
            try:
                response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0, max_tokens=20)
                self.record_usage(response.usage, step="ocr_predict")
                content = response.choices[0].message.content
                raw_texts.append(content.strip() if content is not None else "")
            except Exception as e:
                print(f"Viga OpenRouter API kutsumisel pildi jaoks partiis: {e}")
                raw_texts.append(None) # Näita selle pildi puhul ebaõnnestumist
        return raw_texts

    def get_usage_totals(self) -> dict:
        """Tagastab kogutud kasutusstatistikad."""
        return self.usage_totals

    def get_model_name(self) -> str:
        """Tagastab kasutatava mudeli nime."""
        return self.model