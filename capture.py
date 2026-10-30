import cv2
import os
import re
import time

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "raw")
os.makedirs(SAVE_DIR, exist_ok=True)

SURFACES   = ["asphalt", "concrete", "brick", "gravel"]
CONDITIONS = ["dry", "wet"]

# Site/session tag goes into the filename so dataset splitting can keep
# images from the same location together (avoids train/test leakage).
SITE = "siteA"
# Folder structure:  dataset/raw/<surface>/<condition>/
#   e.g. dataset/raw/asphalt/dry, dataset/raw/asphalt/wet, ...
for _s in SURFACES:
    for _c in CONDITIONS:
        os.makedirs(os.path.join(SAVE_DIR, _s, _c), exist_ok=True)

sel_surface   = 0
sel_condition = 0


def class_dir(surface, condition):
    return os.path.join(SAVE_DIR, surface, condition)


def count_existing(surface, condition):
    folder = class_dir(surface, condition)
    return len([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])


# V4L2 backend + MJPG avoids the GStreamer "internal data stream error" on the Pi
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise SystemExit("Could not open the camera (index 0).")

cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

# ── Button layout (compact: 640 x 420 total, fits small Pi screens) ──────────
CANVAS_W = 640
CAM_H    = 300   # camera preview height
PANEL_H  = 120   # button panel height
BTN_W    = 150
BTN_H    = 48
GAP      = 7

# Row 1: four surface buttons
SRF_BTN = []
for i, s in enumerate(SURFACES):
    x = 10 + i * (BTN_W + GAP)
    SRF_BTN.append({"label": s.upper(), "x": x, "y": CAM_H + 8, "w": BTN_W, "h": BTN_H})

# Row 2: two condition buttons
COND_BTN = []
for i, c in enumerate(CONDITIONS):
    x = 10 + i * (BTN_W + GAP)
    COND_BTN.append({"label": c.upper(), "x": x, "y": CAM_H + 8 + BTN_H + GAP + 2, "w": BTN_W, "h": BTN_H})

# Row 2: capture button, right of the condition buttons
_cap_x = 10 + 2 * (BTN_W + GAP)
CAP_BTN = {"label": "CAPTURE", "x": _cap_x, "y": CAM_H + 8 + BTN_H + GAP + 2,
           "w": CANVAS_W - 10 - _cap_x, "h": BTN_H}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_btn(img, btn, active=False, is_capture=False):
    x, y, w, h = btn["x"], btn["y"], btn["w"], btn["h"]
    if is_capture:
        colour = (40, 160, 60)
    elif active:
        colour = (50, 140, 220)
    else:
        colour = (70, 70, 70)
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    (tw, th), _ = cv2.getTextSize(btn["label"], FONT, 0.55, 2)
    # Capture button label sits a bit higher to leave room for the counter text
    ty = y + (h + th) // 2 - (8 if is_capture else 0)
    cv2.putText(img, btn["label"], (x + (w - tw) // 2, ty),
                FONT, 0.55, (255, 255, 255), 2)


def in_btn(btn, px, py):
    return btn["x"] <= px <= btn["x"] + btn["w"] and btn["y"] <= py <= btn["y"] + btn["h"]


flash_text  = ""
flash_until = 0
last_frame  = None


def on_mouse(event, px, py, flags, param):
    global sel_surface, sel_condition, flash_text, flash_until
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for i, b in enumerate(SRF_BTN):
        if in_btn(b, px, py):
            sel_surface = i
            return

    for i, b in enumerate(COND_BTN):
        if in_btn(b, px, py):
            sel_condition = i
            return

    if in_btn(CAP_BTN, px, py):
        frame = last_frame
        if frame is not None:
            surface   = SURFACES[sel_surface]
            condition = CONDITIONS[sel_condition]
            count     = count_existing(surface, condition) + 1
            filename  = f"{surface}_{condition}_{SITE}_{count:03d}.jpg"
            filepath  = os.path.join(class_dir(surface, condition), filename)
            if cv2.imwrite(filepath, frame):
                flash_text  = f"Saved: {surface}/{condition}/{filename}"
                flash_until = time.time() + 1.2
                print(f"  Saved: {filepath}")
            else:
                print(f"  FAILED to save: {filepath}")


cv2.setMouseCallback("Capture", on_mouse)

print(f"\nSite: {SITE}")
print("Click surface type and condition, then click CAPTURE.")
print("Press Q to quit.\n")

while True:
    ok, frame = cap.read()
    if not ok:
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break
        continue
    last_frame = frame.copy()

    # Full canvas: camera preview on top, button panel below
    canvas = cv2.copyMakeBorder(
        cv2.resize(frame, (CANVAS_W, CAM_H)),
        0, PANEL_H, 0, 0,
        cv2.BORDER_CONSTANT, value=(30, 30, 30)
    )

    for i, b in enumerate(SRF_BTN):
        draw_btn(canvas, b, active=(i == sel_surface))

    for i, b in enumerate(COND_BTN):
        draw_btn(canvas, b, active=(i == sel_condition))

    draw_btn(canvas, CAP_BTN, is_capture=True)

    # Saved-count text inside the capture button
    surface   = SURFACES[sel_surface]
    condition = CONDITIONS[sel_condition]
    count     = count_existing(surface, condition)
    label     = f"{surface}_{condition}: {count} saved"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.4, 1)
    cv2.putText(canvas, label,
                (CAP_BTN["x"] + (CAP_BTN["w"] - tw) // 2,
                 CAP_BTN["y"] + CAP_BTN["h"] - 5),
                FONT, 0.4, (200, 255, 200), 1)

    # Flash message after saving
    if time.time() < flash_until:
        cv2.rectangle(canvas, (0, CAM_H - 30), (CANVAS_W, CAM_H), (0, 150, 60), -1)
        (tw, th), _ = cv2.getTextSize(flash_text, FONT, 0.5, 1)
        cv2.putText(canvas, flash_text, ((CANVAS_W - tw) // 2, CAM_H - 10),
                    FONT, 0.5, (255, 255, 255), 1)

    cv2.imshow("Capture", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")