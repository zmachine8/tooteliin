import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import threading


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
            if not self.running:
                break
            with self.lock:
                self.ret = ret
                self.frame = frame
            if not ret:
                break

    def read(self):
        with self.lock:
            if self.frame is None:
                return self.ret, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


"""
TÖÖ ENNE SEMINARI 3: Liikumise tuvastamine ja viivitusega salvestamine.

Eesmärk:
- Tuvastada konveieril liikumine
- Oodata pildi stabiliseerumist
- Salvestada kaader
- Analüüsida üle lävendi olevate liikumisepisoodide kestust ja vahekaugusi
"""

# --- KONFIGURATSIOON ---
STREAM_URL = "rtsp://172.17.37.81:8554/salami"
# STREAM_URL = "rtsp://172.17.37.81:8554/veis"
# STREAM_URL = "rtsp://172.17.37.81:8554/kalkun"
# STREAM_URL = "rtsp://172.17.37.81:8554/rulaad"

MOTION_THRESHOLD = 18.0          # above this we consider motion detected
STABLE_THRESHOLD = 3.5           # below this we consider frame-to-frame change low
CAPTURE_DELAY = 3.0              # minimum wait after motion start
STABLE_TIME_REQUIRED = 0.7       # how long MAE must stay low before capture
FRAME_SLEEP = 0.02               # pause between two reads

from datetime import datetime
base_folder = STREAM_URL.split('/')[-1]
date_str = datetime.now().strftime("%Y-%m-%d")
report_folder = os.path.join("reports", f"{base_folder}_{date_str}")
os.makedirs(report_folder, exist_ok=True)
images_folder = os.path.join(report_folder, "captures")
os.makedirs(images_folder, exist_ok=True)


def is_green_screen(frame):
    """Detect green start/end screen."""
    if frame is None:
        return False
    small = cv2.resize(frame, (64, 64))
    avg_color = np.mean(small, axis=(0, 1))
    return avg_color[1] > 200 and avg_color[0] < 50 and avg_color[2] < 50


def measure_change(f1, f2):
    """
    Compute frame difference using grayscale + downscale + MAE.
    Returns both the change score and the pure comparison time in milliseconds.
    """
    t0 = time.perf_counter()

    if f1 is None or f2 is None:
        return 0.0, 0.0

    small1 = cv2.resize(f1, (160, 120))
    small2 = cv2.resize(f2, (160, 120))

    gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)

    score = float(np.mean(cv2.absdiff(gray1, gray2)))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"measure_change: {elapsed_ms:.3f} ms | score={score:.3f}")

    return score, elapsed_ms


def summarize_ms(values):
    if len(values) == 0:
        return None

    arr = np.array(values, dtype=float)
    mean_ms = float(np.mean(arr))
    min_ms = float(np.min(arr))
    max_ms = float(np.max(arr))
    median_ms = float(np.median(arr))
    p95_ms = float(np.percentile(arr, 95))

    return {
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
    }


def build_timing_stats(entries):
    if not entries:
        return None

    compare_stats = summarize_ms([entry["compare_time_ms"] for entry in entries])
    loop_stats = summarize_ms([entry["loop_time_ms"] for entry in entries])
    read_stats = summarize_ms([entry["read_time_ms"] for entry in entries])
    sleep_stats = summarize_ms([entry["sleep_time_ms"] for entry in entries])

    return {
        "count": len(entries),
        "compare": compare_stats,
        "loop": loop_stats,
        "read": read_stats,
        "sleep": sleep_stats,
        "theoretical_compare_fps_from_mean": (1000.0 / compare_stats["mean_ms"]) if compare_stats and compare_stats["mean_ms"] > 0 else 0.0,
        "actual_loop_fps_from_mean": (1000.0 / loop_stats["mean_ms"]) if loop_stats and loop_stats["mean_ms"] > 0 else 0.0,
    }


def save_compare_timing_report(entries, path, window_seconds=None):
    if window_seconds is None:
        filtered = entries
        title = "Frame comparison timing report"
        subtitle = "Entire cycle"
    else:
        filtered = [entry for entry in entries if entry["rel_time"] <= window_seconds]
        title = "Frame comparison timing report"
        subtitle = f"First {window_seconds:.0f} seconds"

    stats = build_timing_stats(filtered)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"{'=' * len(title)}\n\n")
        f.write(f"Window: {subtitle}\n\n")

        if not stats:
            f.write("No frame comparison measurements available in this window.\n")
            return

        f.write("Summary\n")
        f.write("-------\n")
        f.write(f"Compared frame pairs:                 {stats['count']}\n")

        f.write("\nMeasured stages\n")
        f.write("---------------\n")
        f.write("Compare time (measure_change only)\n")
        f.write(f"  Average:                            {stats['compare']['mean_ms']:.3f} ms\n")
        f.write(f"  Median:                             {stats['compare']['median_ms']:.3f} ms\n")
        f.write(f"  95th percentile:                    {stats['compare']['p95_ms']:.3f} ms\n")
        f.write(f"  Minimum:                            {stats['compare']['min_ms']:.3f} ms\n")
        f.write(f"  Maximum:                            {stats['compare']['max_ms']:.3f} ms\n")
        f.write(f"  Theoretical FPS from average:       {stats['theoretical_compare_fps_from_mean']:.2f}\n")

        f.write("\nLoop time (one full analysis iteration)\n")
        f.write(f"  Average:                            {stats['loop']['mean_ms']:.3f} ms\n")
        f.write(f"  Median:                             {stats['loop']['median_ms']:.3f} ms\n")
        f.write(f"  95th percentile:                    {stats['loop']['p95_ms']:.3f} ms\n")
        f.write(f"  Minimum:                            {stats['loop']['min_ms']:.3f} ms\n")
        f.write(f"  Maximum:                            {stats['loop']['max_ms']:.3f} ms\n")
        f.write(f"  Actual analysis FPS from average:   {stats['actual_loop_fps_from_mean']:.2f}\n")

        f.write("\nRead time (stream.read calls only)\n")
        f.write(f"  Average:                            {stats['read']['mean_ms']:.3f} ms\n")
        f.write(f"  Median:                             {stats['read']['median_ms']:.3f} ms\n")
        f.write(f"  95th percentile:                    {stats['read']['p95_ms']:.3f} ms\n")
        f.write(f"  Minimum:                            {stats['read']['min_ms']:.3f} ms\n")
        f.write(f"  Maximum:                            {stats['read']['max_ms']:.3f} ms\n")

        f.write("\nRequested sleep between reads\n")
        f.write(f"  Average:                            {stats['sleep']['mean_ms']:.3f} ms\n")
        f.write(f"  Median:                             {stats['sleep']['median_ms']:.3f} ms\n")
        f.write(f"  95th percentile:                    {stats['sleep']['p95_ms']:.3f} ms\n")
        f.write(f"  Minimum:                            {stats['sleep']['min_ms']:.3f} ms\n")
        f.write(f"  Maximum:                            {stats['sleep']['max_ms']:.3f} ms\n")

        f.write("\nInterpretation\n")
        f.write("--------------\n")
        f.write("- Compare time measures only the measure_change() step.\n")
        f.write("- Loop time measures one full analysis iteration in the main while-loop.\n")
        f.write("- Theoretical FPS from average refers only to the comparison step, not the whole program.\n")
        f.write("- Actual analysis FPS from average is based on the full loop time and is the better estimate for end-to-end speed in this script.\n")

        f.write("\nFirst measurements in window\n")
        f.write("----------------------------\n")
        preview = filtered[:20]
        for i, entry in enumerate(preview, start=1):
            f.write(
                f"{i:02d}. rel_time={entry['rel_time']:.3f} s | "
                f"compare_time={entry['compare_time_ms']:.3f} ms | "
                f"loop_time={entry['loop_time_ms']:.3f} ms | "
                f"read_time={entry['read_time_ms']:.3f} ms | "
                f"sleep_time={entry['sleep_time_ms']:.3f} ms | "
                f"change_score={entry['change_score']:.3f}\n"
            )

def save_motion_event_report(events, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Motion event analysis\n")
        f.write("=====================\n\n")
        if not events:
            f.write("No motion events detected above threshold.\n")
            return

        for i, event in enumerate(events):
            f.write(f"Event {i + 1}\n")
            f.write(f"  start_time:    {event['start_time']:.3f} s\n")
            f.write(f"  end_time:      {event['end_time']:.3f} s\n")
            f.write(f"  duration:      {event['duration']:.3f} s\n")
            f.write(f"  peak_score:    {event['peak_score']:.3f}\n")
            f.write(f"  mean_score:    {event['mean_score']:.3f}\n")
            if event["gap_from_previous"] is None:
                f.write("  gap_from_prev: N/A\n")
            else:
                f.write(f"  gap_from_prev: {event['gap_from_previous']:.3f} s\n")
            f.write("\n")


stream = RTSPStreamReader(STREAM_URL)
time.sleep(2)

if not stream.ret:
    print(f"Viga ühendusega: {STREAM_URL}")
    raise SystemExit(1)

print(
    f"Seadistatud: motion threshold={MOTION_THRESHOLD}, "
    f"stable threshold={STABLE_THRESHOLD}, "
    f"capture delay={CAPTURE_DELAY}s, stable time={STABLE_TIME_REQUIRED}s"
)

# --- LOGID GRAAFIKU JA ANALÜÜSI JAOKS ---
timestamps = []
change_scores = []
compare_timing_entries = []

trigger_times = []
capture_times = []
event_start_times = []
event_end_times = []

motion_events = []

started = False
green_cooldown = False
cycle_start_time = 0.0
frame_count = 0
capture_index = 0

# --- motion state ---
in_motion_event = False
current_event_start = None
current_event_scores = []
current_event_peak = 0.0
previous_event_end = None

# --- capture state ---
capture_armed = False
capture_ready_time = None
stable_since = None

try:
    while True:
        loop_start = time.perf_counter()

        read1_start = time.perf_counter()
        ret1, frame1 = stream.read()
        read1_ms = (time.perf_counter() - read1_start) * 1000.0

        sleep_start = time.perf_counter()
        time.sleep(FRAME_SLEEP)
        sleep_ms = (time.perf_counter() - sleep_start) * 1000.0

        read2_start = time.perf_counter()
        ret2, frame2 = stream.read()
        read2_ms = (time.perf_counter() - read2_start) * 1000.0
        read_time_ms = read1_ms + read2_ms

        if not ret1 or not ret2:
            break

        now = time.time()
        current_is_green = is_green_screen(frame2)

        if not started:
            if current_is_green:
                print(">>> Start green detected. Starting cycle.")
                started = True
                green_cooldown = True
                cycle_start_time = now
            continue

        if not current_is_green:
            green_cooldown = False

        if current_is_green and not green_cooldown:
            print(">>> End green detected. Stopping cycle.")
            break

        change, compare_time_ms = measure_change(frame1, frame2)
        rel_time = now - cycle_start_time if cycle_start_time else 0.0

        timestamps.append(rel_time)
        change_scores.append(change)
        loop_time_ms = (time.perf_counter() - loop_start) * 1000.0

        compare_timing_entries.append({
            "rel_time": rel_time,
            "compare_time_ms": compare_time_ms,
            "loop_time_ms": loop_time_ms,
            "read_time_ms": read_time_ms,
            "sleep_time_ms": sleep_ms,
            "change_score": change,
        })
        frame_count += 1

        # -------------------------------------------------
        # 1) Detect above-threshold motion events
        # -------------------------------------------------
        if change > MOTION_THRESHOLD:
            if not in_motion_event:
                in_motion_event = True
                current_event_start = rel_time
                current_event_scores = [change]
                current_event_peak = change
                event_start_times.append(rel_time)

                print(
                    f">>> Motion event START at {rel_time:.3f}s "
                    f"(score={change:.3f})"
                )

                # arm capture only at the first start of a new event
                capture_armed = True
                capture_ready_time = now + CAPTURE_DELAY
                stable_since = None
                trigger_times.append(rel_time)
            else:
                current_event_scores.append(change)
                if change > current_event_peak:
                    current_event_peak = change

        else:
            if in_motion_event:
                # event ends when signal drops back below threshold
                in_motion_event = False
                current_event_end = rel_time
                event_end_times.append(rel_time)

                duration = current_event_end - current_event_start
                mean_score = float(np.mean(current_event_scores)) if current_event_scores else 0.0
                gap_from_previous = (
                    None if previous_event_end is None
                    else current_event_start - previous_event_end
                )

                event_info = {
                    "start_time": current_event_start,
                    "end_time": current_event_end,
                    "duration": duration,
                    "peak_score": current_event_peak,
                    "mean_score": mean_score,
                    "gap_from_previous": gap_from_previous,
                }
                motion_events.append(event_info)

                print(
                    f">>> Motion event END at {current_event_end:.3f}s | "
                    f"duration={duration:.3f}s | "
                    f"peak={current_event_peak:.3f} | "
                    f"mean={mean_score:.3f} | "
                    f"gap_from_previous="
                    f"{'N/A' if gap_from_previous is None else f'{gap_from_previous:.3f}s'}"
                )

                previous_event_end = current_event_end
                current_event_start = None
                current_event_scores = []
                current_event_peak = 0.0

        # -------------------------------------------------
        # 2) Better capture logic:
        #    - wait CAPTURE_DELAY after event start
        #    - then require low MAE for STABLE_TIME_REQUIRED
        # -------------------------------------------------
        if capture_armed and now >= capture_ready_time:
            if change < STABLE_THRESHOLD:
                if stable_since is None:
                    stable_since = now
                elif (now - stable_since) >= STABLE_TIME_REQUIRED:
                    ret_cap, capture_frame = stream.read()
                    if ret_cap and capture_frame is not None:
                        filename = os.path.join(
                            images_folder,
                            f"capture_{capture_index:03d}.jpg"
                        )
                        cv2.imwrite(filename, capture_frame)
                        capture_times.append(rel_time)
                        print(
                            f">>> Captured stable frame: {filename} "
                            f"at {rel_time:.3f}s"
                        )
                        capture_index += 1

                    capture_armed = False
                    capture_ready_time = None
                    stable_since = None
            else:
                stable_since = None

finally:
    stream.stop()

    # If file ends while still inside motion event, close it logically
    if in_motion_event and current_event_start is not None and timestamps:
        current_event_end = timestamps[-1]
        event_end_times.append(current_event_end)

        duration = current_event_end - current_event_start
        mean_score = float(np.mean(current_event_scores)) if current_event_scores else 0.0
        gap_from_previous = (
            None if previous_event_end is None
            else current_event_start - previous_event_end
        )

        motion_events.append({
            "start_time": current_event_start,
            "end_time": current_event_end,
            "duration": duration,
            "peak_score": current_event_peak,
            "mean_score": mean_score,
            "gap_from_previous": gap_from_previous,
        })

    # save text summaries
    report_path = os.path.join(report_folder, "motion_events.txt")
    save_motion_event_report(motion_events, report_path)

    compare_report_60s_path = os.path.join(report_folder, "frame_compare_report_first_60s.txt")
    save_compare_timing_report(compare_timing_entries, compare_report_60s_path, window_seconds=60.0)

    # graph
    plt.figure(figsize=(12, 6))

    if timestamps and change_scores:
        plt.plot(timestamps, change_scores, label="Change score (MAE)")
        plt.axhline(
            y=MOTION_THRESHOLD,
            linestyle="--",
            label=f"Motion threshold = {MOTION_THRESHOLD}"
        )
        plt.axhline(
            y=STABLE_THRESHOLD,
            linestyle=":",
            label=f"Stable threshold = {STABLE_THRESHOLD}"
        )

        for i, t in enumerate(event_start_times):
            label = "Motion event start" if i == 0 else None
            plt.axvline(x=t, linestyle=":", alpha=0.8, label=label)

        for i, t in enumerate(event_end_times):
            label = "Motion event end" if i == 0 else None
            plt.axvline(x=t, linestyle="--", alpha=0.5, label=label)

        for i, t in enumerate(capture_times):
            label = "Captured image" if i == 0 else None
            plt.axvline(x=t, linestyle="-.", alpha=0.8, label=label)

    plt.title("Motion graph over time")
    plt.xlabel("Time from cycle start (s)")
    plt.ylabel("Change score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    graph_path = os.path.join(report_folder, "liikumise_graafik.png")
    plt.savefig(graph_path)
    plt.close()

    print(f"Graph saved: {graph_path}")
    print(f"Motion event report saved: {report_path}")
    print(f"First 60s compare report saved: {compare_report_60s_path}")
    print(f"Compared frame pairs: {frame_count}")
    print(f"Saved images: {capture_index}")

    first_60s_timing_stats = build_timing_stats(
        [entry for entry in compare_timing_entries if entry["rel_time"] <= 60.0]
    )

    if first_60s_timing_stats:
        print("\nFrame comparison timing summary (first 60 seconds):")
        print(f"  Compared frame pairs: {first_60s_timing_stats['count']}")
        print(f"  Compare average: {first_60s_timing_stats['compare']['mean_ms']:.3f} ms")
        print(f"  Compare median:  {first_60s_timing_stats['compare']['median_ms']:.3f} ms")
        print(f"  Compare P95:     {first_60s_timing_stats['compare']['p95_ms']:.3f} ms")
        print(f"  Compare min:     {first_60s_timing_stats['compare']['min_ms']:.3f} ms")
        print(f"  Compare max:     {first_60s_timing_stats['compare']['max_ms']:.3f} ms")
        print(f"  Theoretical compare FPS from average: {first_60s_timing_stats['theoretical_compare_fps_from_mean']:.2f}")
        print(f"  Loop average:    {first_60s_timing_stats['loop']['mean_ms']:.3f} ms")
        print(f"  Loop median:     {first_60s_timing_stats['loop']['median_ms']:.3f} ms")
        print(f"  Actual loop FPS from average: {first_60s_timing_stats['actual_loop_fps_from_mean']:.2f}")

    if motion_events:
        print("\nMotion event summary:")
        for i, event in enumerate(motion_events, start=1):
            gap_text = "N/A" if event["gap_from_previous"] is None else f"{event['gap_from_previous']:.3f}s"
            print(
                f"  Event {i}: "
                f"start={event['start_time']:.3f}s, "
                f"end={event['end_time']:.3f}s, "
                f"duration={event['duration']:.3f}s, "
                f"peak={event['peak_score']:.3f}, "
                f"mean={event['mean_score']:.3f}, "
                f"gap_from_previous={gap_text}"
            )
    else:
        print("No motion events detected above threshold.")