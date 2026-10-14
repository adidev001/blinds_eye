"""
Blind's Eye — Localization & Announcement Formatter
=====================================================
Provides zero-dependency translation of COCO object labels, directions,
and distance strings into Hindi (Devanagari) and Hinglish (Roman script)
for use by AnnouncementTracker and TTSEngine.

Design rules
------------
* This module has NO imports from the rest of the project.
* All dicts are module-level constants — no file I/O, no network calls.
* Falls back to the original English string for any untranslated entry.
* Strings are phoneme-safe for:
    - Piper TTS  ``hi_IN`` models (requires Devanagari for ``--lang hi``)
    - Kokoro-82M ``lang='hi'``    (handles both Devanagari and Roman)
    - Kokoro-82M ``lang='en-us'`` (used for Hinglish Roman strings)
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════
# Supported languages
# ══════════════════════════════════════════════════════════════════════════

SUPPORTED_LANGS: frozenset[str] = frozenset({"en", "hi", "hinglish"})


# ══════════════════════════════════════════════════════════════════════════
# COCO-80 Label Translations
# ══════════════════════════════════════════════════════════════════════════

# fmt: off
COCO_LABELS_HI: dict[str, str] = {
    # People & Vehicles
    "person":           "व्यक्ति",
    "bicycle":          "साइकिल",
    "car":              "कार",
    "motorcycle":       "मोटरसाइकिल",
    "airplane":         "हवाई जहाज़",
    "bus":              "बस",
    "train":            "ट्रेन",
    "truck":            "ट्रक",
    "boat":             "नाव",
    # Road objects
    "traffic light":    "ट्रैफ़िक लाइट",
    "fire hydrant":     "फायर हाइड्रेंट",
    "stop sign":        "स्टॉप साइन",
    "parking meter":    "पार्किंग मीटर",
    "bench":            "बेंच",
    # Animals
    "bird":             "पक्षी",
    "cat":              "बिल्ली",
    "dog":              "कुत्ता",
    "horse":            "घोड़ा",
    "sheep":            "भेड़",
    "cow":              "गाय",
    "elephant":         "हाथी",
    "bear":             "भालू",
    "zebra":            "ज़ेब्रा",
    "giraffe":          "जिराफ़",
    # Accessories
    "backpack":         "बैगपैक",
    "umbrella":         "छाता",
    "handbag":          "हैंडबैग",
    "tie":              "टाई",
    "suitcase":         "सूटकेस",
    # Sports
    "frisbee":          "फ्रिसबी",
    "skis":             "स्की",
    "snowboard":        "स्नोबोर्ड",
    "sports ball":      "खेल की गेंद",
    "kite":             "पतंग",
    "baseball bat":     "बेसबॉल बैट",
    "baseball glove":   "बेसबॉल दस्ताना",
    "skateboard":       "स्केटबोर्ड",
    "surfboard":        "सर्फ़बोर्ड",
    "tennis racket":    "टेनिस रैकेट",
    # Kitchen / Food
    "bottle":           "बोतल",
    "wine glass":       "वाइन ग्लास",
    "cup":              "कप",
    "fork":             "काँटा",
    "knife":            "चाकू",
    "spoon":            "चम्मच",
    "bowl":             "कटोरा",
    "banana":           "केला",
    "apple":            "सेब",
    "sandwich":         "सैंडविच",
    "orange":           "संतरा",
    "broccoli":         "ब्रोकली",
    "carrot":           "गाजर",
    "hot dog":          "हॉट डॉग",
    "pizza":            "पिज़्ज़ा",
    "donut":            "डोनट",
    "cake":             "केक",
    # Furniture / Indoor
    "chair":            "कुर्सी",
    "couch":            "सोफ़ा",
    "potted plant":     "गमले का पौधा",
    "bed":              "बिस्तर",
    "dining table":     "खाने की मेज़",
    "toilet":           "शौचालय",
    # Electronics
    "tv":               "टेलीविज़न",
    "laptop":           "लैपटॉप",
    "mouse":            "माउस",
    "remote":           "रिमोट",
    "keyboard":         "कीबोर्ड",
    "cell phone":       "मोबाइल फ़ोन",
    "microwave":        "माइक्रोवेव",
    "oven":             "ओवन",
    "toaster":          "टोस्टर",
    "sink":             "सिंक",
    "refrigerator":     "फ्रिज",
    # Misc
    "book":             "किताब",
    "clock":            "घड़ी",
    "vase":             "फूलदान",
    "scissors":         "कैंची",
    "teddy bear":       "टेडी बियर",
    "hair drier":       "हेयर ड्रायर",
    "toothbrush":       "टूथब्रश",
}

COCO_LABELS_HINGLISH: dict[str, str] = {
    # People & Vehicles
    "person":           "insaan",
    "bicycle":          "cycle",
    "car":              "car",
    "motorcycle":       "bike",
    "airplane":         "havaai jahaz",
    "bus":              "bus",
    "train":            "train",
    "truck":            "truck",
    "boat":             "naav",
    # Road objects
    "traffic light":    "traffic light",
    "fire hydrant":     "fire hydrant",
    "stop sign":        "stop sign",
    "parking meter":    "parking meter",
    "bench":            "bench",
    # Animals
    "bird":             "pakshi",
    "cat":              "billi",
    "dog":              "kutta",
    "horse":            "ghoda",
    "sheep":            "bhed",
    "cow":              "gaay",
    "elephant":         "haathi",
    "bear":             "bhaalu",
    "zebra":            "zebra",
    "giraffe":          "giraffe",
    # Accessories
    "backpack":         "bag",
    "umbrella":         "chhaata",
    "handbag":          "handbag",
    "tie":              "tie",
    "suitcase":         "suitcase",
    # Sports
    "frisbee":          "frisbee",
    "skis":             "ski",
    "snowboard":        "snowboard",
    "sports ball":      "ball",
    "kite":             "patang",
    "baseball bat":     "bat",
    "baseball glove":   "glove",
    "skateboard":       "skateboard",
    "surfboard":        "surfboard",
    "tennis racket":    "racket",
    # Kitchen / Food
    "bottle":           "bottle",
    "wine glass":       "glass",
    "cup":              "cup",
    "fork":             "kaanta",
    "knife":            "chaaqu",
    "spoon":            "chamach",
    "bowl":             "katora",
    "banana":           "kela",
    "apple":            "seb",
    "sandwich":         "sandwich",
    "orange":           "santra",
    "broccoli":         "broccoli",
    "carrot":           "gajar",
    "hot dog":          "hot dog",
    "pizza":            "pizza",
    "donut":            "donut",
    "cake":             "cake",
    # Furniture / Indoor
    "chair":            "kursi",
    "couch":            "sofa",
    "potted plant":     "plant",
    "bed":              "bistar",
    "dining table":     "table",
    "toilet":           "toilet",
    # Electronics
    "tv":               "TV",
    "laptop":           "laptop",
    "mouse":            "mouse",
    "remote":           "remote",
    "keyboard":         "keyboard",
    "cell phone":       "mobile",
    "microwave":        "microwave",
    "oven":             "oven",
    "toaster":          "toaster",
    "sink":             "sink",
    "refrigerator":     "fridge",
    # Misc
    "book":             "kitaab",
    "clock":            "ghadi",
    "vase":             "phooldan",
    "scissors":         "kainchi",
    "teddy bear":       "teddy",
    "hair drier":       "hair dryer",
    "toothbrush":       "toothbrush",
}
# fmt: on


# ══════════════════════════════════════════════════════════════════════════
# Direction Translations
# ══════════════════════════════════════════════════════════════════════════

DIRECTIONS_HI: dict[str, str] = {
    "left":   "बाईं ओर",
    "center": "सामने",
    "right":  "दाईं ओर",
}

DIRECTIONS_HINGLISH: dict[str, str] = {
    "left":   "baayein",
    "center": "saamne",
    "right":  "daayein",
}


# ══════════════════════════════════════════════════════════════════════════
# Announcement Sentence Templates
# ══════════════════════════════════════════════════════════════════════════
# Slots:
#   {label}  — translated object name
#   {dir}    — translated direction
#   {dist}   — distance as a numeric string (e.g. "2.1")
#   {item}   — single formatted detection string
#   {items}  — comma-joined detection strings

TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "with_depth":   "{label} on the {dir} at {dist} meters",
        "no_depth":     "{label} on the {dir}",
        "single":       "I see {item}",
        "multi":        "I see: {items}",
        # System messages — spoken at startup / shutdown
        "startup":      "Blind's Eye is starting up.",
        "cam_ready":    "Camera ready. Beginning detection.",
        "shutdown":     "Shutting down. Goodbye.",
        "cam_error":    "Error: camera not available.",
    },
    "hi": {
        "with_depth":   "{dir} {label}, {dist} मीटर पर",
        "no_depth":     "{dir} {label}",
        "single":       "मैं देख रहा हूँ {item}",
        "multi":        "मैं देख रहा हूँ: {items}",
        "startup":      "Blind's Eye शुरू हो रहा है।",
        "cam_ready":    "कैमरा तैयार है। पहचान शुरू।",
        "shutdown":     "बंद हो रहा है। नमस्ते।",
        "cam_error":    "त्रुटि: कैमरा उपलब्ध नहीं।",
    },
    "hinglish": {
        "with_depth":   "{dir} {label}, {dist} meter par",
        "no_depth":     "{dir} {label} hai",
        "single":       "dekh raha hoon {item}",
        "multi":        "dekh raha hoon: {items}",
        "startup":      "Blind's Eye shuru ho raha hai.",
        "cam_ready":    "Camera taiyar. Detection shuru.",
        "shutdown":     "Band ho raha hai. Alvida.",
        "cam_error":    "Error: camera nahi mila.",
    },
}


# ══════════════════════════════════════════════════════════════════════════
# LocalizationFormatter
# ══════════════════════════════════════════════════════════════════════════

class LocalizationFormatter:
    """
    Translates COCO detection strings and directions into the requested
    language and builds ready-to-speak announcement sentences.

    Parameters
    ----------
    lang : str
        One of ``"en"``, ``"hi"``, ``"hinglish"``.

    Examples
    --------
    >>> fmt = LocalizationFormatter("hi")
    >>> fmt.format_detection("person", "left", 2.1)
    'बाईं ओर व्यक्ति, 2.1 मीटर पर'
    >>> fmt.format_announcement(["बाईं ओर व्यक्ति, 2.1 मीटर पर"])
    'मैं देख रहा हूँ बाईं ओर व्यक्ति, 2.1 मीटर पर'
    """

    def __init__(self, lang: str = "en") -> None:
        if lang not in SUPPORTED_LANGS:
            raise ValueError(
                f"Unsupported lang: {lang!r}. Must be one of {sorted(SUPPORTED_LANGS)}"
            )
        self.lang = lang
        self._labels: dict[str, str] | None = (
            COCO_LABELS_HI       if lang == "hi"
            else COCO_LABELS_HINGLISH if lang == "hinglish"
            else None
        )
        self._dirs: dict[str, str] | None = (
            DIRECTIONS_HI        if lang == "hi"
            else DIRECTIONS_HINGLISH  if lang == "hinglish"
            else None
        )
        self._tmpl = TEMPLATES[lang]

    # ── Translation helpers ────────────────────────────────────────────────

    def translate_label(self, label: str) -> str:
        """Return the translated label, falling back to the raw English name."""
        if self._labels is None:
            return label
        return self._labels.get(label, label)

    def translate_direction(self, direction: str) -> str:
        """Return the translated direction word."""
        if self._dirs is None:
            return direction
        return self._dirs.get(direction, direction)

    # ── Sentence builders ─────────────────────────────────────────────────

    def format_detection(
        self,
        label: str,
        direction: str,
        depth_meters: float | None,
    ) -> str:
        """
        Build a single-object detection sentence.

        Parameters
        ----------
        label       : COCO English label string
        direction   : ``"left"`` | ``"center"`` | ``"right"``
        depth_meters: metric depth float, or ``None`` if depth is disabled
        """
        l = self.translate_label(label)
        d = self.translate_direction(direction)
        if depth_meters is not None:
            return self._tmpl["with_depth"].format(
                label=l, dir=d, dist=f"{depth_meters:.1f}"
            )
        return self._tmpl["no_depth"].format(label=l, dir=d)

    def format_announcement(self, items: list[str]) -> str:
        """
        Wrap one or more formatted detection strings into a full sentence.

        Parameters
        ----------
        items : list of formatted detection strings (from ``format_detection``)
        """
        if len(items) == 1:
            return self._tmpl["single"].format(item=items[0])
        return self._tmpl["multi"].format(items=", ".join(items))

    def system_message(self, key: str) -> str:
        """
        Return a startup / shutdown / error message in the target language.

        Parameters
        ----------
        key : one of ``"startup"``, ``"cam_ready"``, ``"shutdown"``, ``"cam_error"``
        """
        return self._tmpl.get(key, TEMPLATES["en"].get(key, ""))
