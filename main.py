"""
VirtualBOT - a voice/text-controlled desktop agent powered by Gemini
====================================================================

Fast explanation:
    1. You speak (hold F4 / in some computers fn+F4) or press ctrl and type a command, for example
       "open Chrome and search for cats".
    2. Gemini transcribes your audio (speech-to-text).
    3. Gemini PLANS: the command is split into sequential atomic steps.
    4. For each step, a screenshot is taken and sent to Gemini, which:
         a) LOCATES the target UI element and returns normalized coordinates
         b) CLASSIFIES the action according to the declared actions
    5. PyAutoGUI executes the action on the real screen.

Requirements:
    pip install google-genai pyautogui mss sounddevice soundfile keyboard numpy
    export GEMINI_API_KEY="your-key"
    -> A GEMINI API KEY IS NEEDED, it can be obtained in Google AI Studio

ATTENTION: this program controls your mouse cursor and keyboard.

Keep the target window in the foreground and don't touch anything while a
command runs. Say/type "stop" to quit.

It's nice for saving time doing other activities while the VBOT does the boring part
For example, fill the water bottle while it opens the web-browser and multiple tabs
"""

import asyncio
import io
import os
import time
import keyboard as kb  # used for global hotkeys
import numpy as np
import re
import json

import pyautogui as pg  # moves the mouse / presses keys
from mss import MSS  # very fast cross-platform screenshots
import mss.tools
import sounddevice as sd  # microphone capture
import soundfile as sf  # encodes raw audio into WAV
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration ("MACROs")
# ---------------------------------------------------------------------------
# A "lite" model keeps latency low. This matters a lot here, since
# every single step makes 2 vision calls, and the bot feels slow otherwise.
CHAT_MODEL = "gemini-3.5-flash-lite"
STT_MODEL = "gemini-3.5-flash-lite"

# Forces Gemini to answer with valid JSON, so we can json.loads() the reply
# without begging the model nicely in the prompt.
JSON_CONFIG = types.GenerateContentConfig(response_mime_type="application/json")

SAMPLE_RATE = 16000  # 16 kHz is a nice frequency for speech recognition
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# By default, slamming the mouse into a screen corner aborts PyAutoGUI.
# It is disabled because the bot legitimately clicks at (0, 0) sometimes.
pg.FAILSAFE = False

# Hotkeys: hold F4 to talk, press Ctrl to type. Changeable with
# the "config" command.
write_key, voice_key = 'ctrl', 'f4'

# ---------------------------------------------------------------------------
# Prompt snippets appended to each user step
# ---------------------------------------------------------------------------

# Gemini returns coordinates normalized to a 0-1000 grid, no matter the real
# screen resolution. That makes the bot resolution-independent: the conversion
# to pixels happens at the last moment (see VirtualBOT.to_pixels).
COORD_INSTRUCTION = (
    "\n\nLocate in the image the element described above. "
    'Answer only with JSON in the shape {"point": [x, y], "label": "element name"}, '
    "where x and y are integers normalized from 0 to 1000 (0,0 = left high edge)."
)
# NOTE: Gemini's native pointing format is actually [y, x]. That is why
# parse_point() below unpacks the list as `y, x`. If you ever get clicks on
# mirrored positions, check this prompt/parser pair first!

# The "planner" prompt: turns "open notepad and write hello" into
# ["open notepad", "write hello"] so each piece can be handled with its own
# screenshot (the screen changes after every step).
INTERPRETER_INSTRUCTION = (
    "\n\nI want you to break the prompt above into different sequential commands if the prompt has multiple orders. "
    'Return ONLY a JSON array of strings representing each step, e.g., ["command 1", "command 2"].'
)

# The "intent classifier" prompt: decides HOW to act on a located element.
# Keeping locate and classify as two separate calls makes each prompt simpler,
# which tends to make a small model more reliable.
DETECT_ACTION_INSTRUCTION = (
    "\n\nAnalyze the user prompt above and return a JSON in this exact shape:\n"
    '{"mouse_click": true, "keyboard_typing": false, "text_to_write": null, "action": "none", "double_click": false}\n'
    "RULES:\n"
    "1. 'action' can be ONLY 'none', 'close', or 'new_tab'.\n"
    "2. IF 'action' is 'new_tab' OR 'close', YOU MUST SET 'mouse_click' TO false AND 'keyboard_typing' TO false.\n"
    "3. Set 'double_click' to true if the step implies opening a desktop application, file, or shortcut by double-clicking.\n"
    "4. Set 'mouse_click' to true ONLY when clicking on a regular element."
)


def enhance(func):
    """Decorator that wraps a function's output between two separator lines.
    Purely cosmetic: makes it easy to see the console output"""
    def enhancer(*args, **kwargs):
        print("=" * 80)
        res = func(*args, **kwargs)
        print("=" * 80)
        return res
    return enhancer


class Agent:
    """Holds the global settings: model used, transcription,
    and the 'spec' / 'config' menus."""

    def __init__(self):
        self.client = client
        self.md = CHAT_MODEL
        self.ans = ""

    @staticmethod
    def config():
        """configure the voice or write hotkey without restarting the program."""
        global write_key, voice_key
        opt = input("Do you want to change voice key or write key? Answer: write or voice\n")
        if opt == "write":
            write_key = input("New write key -> ")
        elif opt == "voice":
            voice_key = input("New voice key -> ")
        return write_key, voice_key

    @enhance
    def spec(self):
        """Prints a quick help screen (model, hotkeys, usage warnings)."""
        print("Model:", self.md)
        print(f"Voice Key: {voice_key}\nWrite Key: {write_key}")
        print(f"For inserting commands by writing, before you must press {write_key}")
        print(f"For using voice recognition, you must hold {voice_key} while speaking")
        print("The program that will be operated by gemini MUST be in PRIMARY use")
        print("IF SOMETHING ELSE WAS CLICKED/SELECTED, THE ASSISTANT MAY NOT WORK PROPERLY")


class Prompt:
    """It initializes some generic variables that will be replaced
    in the main function at the end, and defines some important functions
    for interacting with Gemini"""
    def __init__(self, agent):
        self.client = agent.client
        self.ans = agent.ans
        self.md = agent.md
        self.image_bytes = None
        self.chat = self.client.aio.chats.create(model=self.md)

    def reset_chat(self):
        # A fresh chat per command: old screenshots/steps,
        # sometimes having an old chat loaded drives the
        # script to errors.
        self.chat = self.client.aio.chats.create(model=self.md)

    async def interact(self, text, image_bytes=None, config=None):
        """Send text/screenshot to Gemini and await the reply."""
        contents = [text]
        if image_bytes:
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            )
        return await self.chat.send_message(contents, config=config)

    def output(self, response):
        """Just prints a response."""
        commands = []
        if response is None:
            print("No response received.")
            return None
        print(response.text or "[Null]")
        for cmd in response.text.split("\n"):
            commands.append(cmd)

        if len(commands) == 0:
            print("No valid commands received.")
        return commands


class Recognizer():
    """Push-to-talk speech recognition: record while holding the key,
    then let Gemini transcribe the audio."""

    def __init__(self):
        self.client = client

    def listen(self):
        frames = []

        # sounddevice calls this from a background thread for every audio
        # chunk.
        def callback(indata, frame_count, time_info, status):
            if status:
                print(status)
            frames.append(indata.copy())

        print(f"🔴 Recording... release {voice_key} to stop\n")
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="int16", callback=callback):
            # The stream records for as long as the key pressed.
            while kb.is_pressed(voice_key):
                time.sleep(0.05)

        if not frames:
            raise RuntimeError("No audio captured. Check your microphone.")

        # Puts all chunks together and encode as an WAV file
        # and writes it in the buffer
        audio = np.concatenate(frames, axis=0)
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def transcribe(self, wav_bytes):
        """Where te transcription happens"""
        print("\n[Gemini] Transcribing audio...")
        resp = self.client.models.generate_content(
            model=STT_MODEL,
            contents=[
                "Transcribe the sent audio into text. Return only the transcription.",
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            ],
        )
        text = (resp.text or "").strip()
        print(f"[Transcribed]: {text}")
        return text


class VirtualBOT:
    """The virtual assistant: takes screenshots and performs mouse and keyboard actions."""

    def __init__(self):
        self.px = None  # last target X in real pixels
        self.py = None  # last target Y in real pixels

    @staticmethod
    def prt():
        """Grab the primary monitor and return it as PNG bytes (saved in memory).
        This will be the image that represents this exact moment."""
        with MSS() as sct:
            shot = sct.grab(sct.monitors[1])  # monitors[0] = all screens, [1] = primary (just for cases with more screens)
            return mss.tools.to_png(shot.rgb, shot.size)

    @staticmethod
    def parse_point(text):
        """Turn Gemini's JSON answer into (x, y, label), all normalized 0-1000.

        Gemini sometimes returns JSON in ```json fences even when told not to,
        so the answer is striped. Two answer shapes are supported:
          - {"point": [y, x]}                 -> direct point
          - {"box_2d": [ymin, xmin, ymax, xmax]} -> we click the box center
        """
        text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
        data = json.loads(text)
        if isinstance(data, list):  # sometimes the model returns [ {...} ]
            data = data[0]
        if "point" in data:
            y, x = data["point"]  # Gemini's native order is [y, x]
            return float(x), float(y), data.get("label", "")
        elif "box_2d" in data:
            ymin, xmin, ymax, xmax = data["box_2d"]
            y_center = (ymin + ymax) / 2
            x_center = (xmin + xmax) / 2
            return float(x_center), float(y_center), data.get("label", "")
        else:
            raise KeyError("There was not found either a point or a box2d")

    def to_pixels(self, x_norm, y_norm):
        """Convert the 0-1000 grid to real screen pixels, clamped to the screen
        so a slightly-off prediction can never send the mouse out of the predicted area."""
        width, height = pg.size()
        self.px = min(max(int((x_norm / 1000) * width), 0), width - 1)
        self.py = min(max(int((y_norm / 1000) * height), 0), height - 1)
        return self.px, self.py

    def click(self, x_norm=0, y_norm=0, text_to_write=None, double_click=False, action=None):
        """Execute one action. Priority order: tab shortcuts -> typing -> double click -> click."""
        self.to_pixels(x_norm, y_norm)

        # Tab actions: focus the window with a click first, then use the
        # keyboard shortcut (faster and more reliable than hunting for the
        # little 'x' and '+' buttons).
        if action == 'close':
            pg.click(self.px, self.py)
            time.sleep(0.2)
            pg.hotkey('ctrl', 'w')
            print("[VirtualBOT] Closing tab.")
            return
        elif action == 'new_tab':
            pg.click(self.px, self.py)
            time.sleep(0.2)
            pg.hotkey('ctrl', 't')
            print("[VirtualBOT] New tab.")
            return

        # Move the cursor smoothly instead of teleporting: it looks better,
        # and it's easier to keep track of the cursor
        pg.moveTo(self.px, self.py, duration=0.3)
        time.sleep(0.2)

        if text_to_write:
            # Click the field, type and press Enter.
            pg.click(self.px, self.py)
            time.sleep(0.3)
            pg.write(str(text_to_write), interval=0.05)
            time.sleep(0.2)
            pg.press('enter')
            print(f"[VirtualBOT] Typed '{text_to_write}' at ({self.px}, {self.py})")
        elif double_click:
            pg.doubleClick(self.px, self.py)
            print(f"[VirtualBOT] Double clicked at ({self.px}, {self.py})")
        else:
            # Manual mouseDown/mouseUp with a small pause instead of pg.click():
            # some apps drop clicks that are too instantaneous.
            pg.mouseDown(self.px, self.py, button='left')
            time.sleep(0.05)
            pg.mouseUp(self.px, self.py, button='left')
            print(f"[VirtualBOT] Single clicked safely at ({self.px}, {self.py})")


# Instantiating the objects
agt = Agent()
ppt = Prompt(agt)
vbot = VirtualBOT()


async def main():
    print("\n--- VirtualBOT ---")
    print(f"Press and hold {voice_key} for speaking or {write_key} for typing.")

    rec = Recognizer()

    # "listen -> plan -> see -> act" cycle.
    while True:
        ppt.reset_chat() # Cleaning the chat

        if kb.is_pressed(voice_key):
            try:
                wav = rec.listen()
                agt.ans = rec.transcribe(wav)
                choice = agt.ans.strip().lower()
            except Exception as e:
                print(f"Error in audio processing: {e}")
                continue
        elif kb.is_pressed(write_key):
            user_input = input("> ")
            choice = user_input.strip().lower()

            # Wait until the key is released, so it works one press per command.
            while kb.is_pressed(write_key):
                time.sleep(0.05)
        else:
            # When nothing is pressed it yields to the event loop and polls again.
            await asyncio.sleep(0.05)
            continue

        # Built-in commands
        if choice == "stop":
            print("Exiting Virtual BOT...")
            break
        elif choice == "spec":
            agt.spec() # Specification
            continue
        elif choice == "config":
            agt.config() # Configs related to the key binds
            continue

        ppt.image_bytes = None
        agt.ans = choice

        try:
            # Planning
            print("[VirtualBOT] Planning steps...")

            plan_response = await ppt.interact(
                f"User command: '{agt.ans}'." + INTERPRETER_INSTRUCTION,
                config=JSON_CONFIG
            )

            raw_text = re.sub(r"^```(?:json)?|```$", "", (plan_response.text or "").strip(), flags=re.M).strip()
            commands_list = json.loads(raw_text)

            # Defensive normalization: LLMs don't always respect the schema.
            if isinstance(commands_list, str):
                commands_list = [commands_list]
            elif isinstance(commands_list, dict):
                commands_list = list(commands_list.values())

            print(f"[VirtualBOT] Planned steps: {commands_list}")

            # See -> Think -> Act, step by step
            for step_item in commands_list:
                if isinstance(step_item, dict):
                    step = step_item.get("command") or step_item.get("step") or str(step_item)
                else:
                    step = str(step_item)

                print(f"\n--- Executing step: '{step}' ---")

                # New screenshot every step, the previous action changed the screen.
                image_bytes = vbot.prt()

                # Locating the target
                fst_response = await ppt.interact(
                    step + COORD_INSTRUCTION, image_bytes, config=JSON_CONFIG
                )
                x, y, label = vbot.parse_point(fst_response.text)

                # Linking it to the command, figuring out how to use it
                scd_response = await ppt.interact(
                    step + DETECT_ACTION_INSTRUCTION, image_bytes, config=JSON_CONFIG
                )
                data = json.loads(scd_response.text)

                mouse_click = data.get("mouse_click", False)
                action = data.get("action", "none")
                keyboard_typing = data.get("keyboard_typing", False)
                text_to_write = data.get("text_to_write")
                double_click = data.get("double_click", False)

                if action in ["new_tab", "close"]:
                    # Lands at (0, 0), the top-left corner. Pass x_norm/y_norm
                    # if you want to focus a specific window first.
                    vbot.click(action=action)
                    print(f"[VirtualBOT] Executed action: {action}")
                elif mouse_click or keyboard_typing or text_to_write:
                    vbot.click(
                        x_norm=x,
                        y_norm=y,
                        text_to_write=text_to_write,
                        double_click=double_click,
                        action="none"
                    )
                # Give the UI time to react (animations, page loads).
                # A function for acknowledging if the page has loaded would be good
                # and probably will be done on an update.
                time.sleep(1.0)

        # Error handling for showing in the logs that something is wrong.
        except json.JSONDecodeError as e:
            print(f"[Error]: No valid JSON, detail: {e}")
        except KeyError as e:
            print(f"[Error]: No coords, detail: {e}")
        except Exception as e:
            print(f"[Unexpected ERROR]: {e}")


if __name__ == "__main__":
    asyncio.run(main())