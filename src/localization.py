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
# Curated COCO Plural Maps (FR-3.3)
# ══════════════════════════════════════════════════════════════════════════

COCO_PLURALS_EN: dict[str, str] = {
    "person":           "people",
    "bicycle":          "bicycles",
    "car":              "cars",
    "motorcycle":       "motorcycles",
    "airplane":         "airplanes",
    "bus":              "buses",
    "train":            "trains",
    "truck":            "trucks",
    "boat":             "boats",
    "traffic light":    "traffic lights",
    "fire hydrant":     "fire hydrants",
    "stop sign":        "stop signs",
    "parking meter":    "parking meters",
    "bench":            "benches",
    "bird":             "birds",
    "cat":              "cats",
    "dog":              "dogs",
    "horse":            "horses",
    "sheep":            "sheep",
    "cow":              "cows",
    "elephant":         "elephants",
    "bear":             "bears",
    "zebra":            "zebras",
    "giraffe":          "giraffes",
    "backpack":         "backpacks",
    "umbrella":         "umbrellas",
    "handbag":          "handbags",
    "tie":              "ties",
    "suitcase":         "suitcases",
    "chair":            "chairs",
    "couch":            "couches",
    "potted plant":     "potted plants",
    "bed":              "beds",
    "dining table":     "dining tables",
    "toilet":           "toilets",
    "tv":               "tvs",
    "laptop":           "laptops",
    "mouse":            "mice",
    "remote":           "remotes",
    "keyboard":         "keyboards",
    "cell phone":       "cell phones",
    "bottle":           "bottles",
    "cup":              "cups",
    "fork":             "forks",
    "knife":            "knives",
    "spoon":            "spoons",
    "bowl":             "bowls",
    "book":             "books",
    "clock":            "clocks",
    "vase":             "vases",
    "scissors":         "scissors",
}

COCO_PLURALS_HI: dict[str, str] = {
    "person":           "लोग",
    "bicycle":          "साइकिलें",
    "car":              "कारें",
    "motorcycle":       "बाइक",
    "airplane":         "हवाई जहाज़",
    "bus":              "बसें",
    "train":            "ट्रेनें",
    "truck":            "ट्रक",
    "boat":             "नावें",
    "traffic light":    "ट्रैफ़िक लाइटें",
    "fire hydrant":     "फायर हाइड्रेंट",
    "stop sign":        "स्टॉप साइन",
    "bench":            "बेंचें",
    "bird":             "पक्षी",
    "cat":              "बिल्लियाँ",
    "dog":              "कुत्ते",
    "horse":            "घोड़े",
    "sheep":            "भेड़ें",
    "cow":              "गायें",
    "elephant":         "हाथी",
    "chair":            "कुर्सियाँ",
    "couch":            "सोफ़े",
    "bed":              "बिस्तर",
    "dining table":     "मेज़ें",
    "bottle":           "बोतलें",
    "cup":              "कप",
    "book":             "किताबें",
    "clock":            "घड़ियाँ",
}

COCO_PLURALS_HINGLISH: dict[str, str] = {
    "person":           "log",
    "bicycle":          "cycles",
    "car":              "gaadiyan",
    "motorcycle":       "bikes",
    "bus":              "basen",
    "train":            "trains",
    "truck":            "trucks",
    "bench":            "benches",
    "cat":              "billiyan",
    "dog":              "kutte",
    "chair":            "kursiyan",
    "couch":            "couches",
    "bed":              "beds",
    "bottle":           "bottles",
    "cup":              "cups",
    "book":             "kitaben",
}


# ══════════════════════════════════════════════════════════════════════════
# Dual-Regime Distance Quantization (FR-3.1)
# ══════════════════════════════════════════════════════════════════════════

def quantize_distance(depth: float) -> float:
    """
    Quantize metric depth into perceptual bins (FR-3.1):
    - < 1.5m : round to nearest 0.1m
    - >= 1.5m: round to nearest 0.5m
    """
    if depth < 1.5:
        return round(float(depth), 1)
    return round(float(depth) * 2.0) / 2.0


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

    def get_plural_label(self, label: str) -> str:
        """Return the curated plural form of *label* in the current language."""
        if self.lang == "en":
            return COCO_PLURALS_EN.get(label, label + "s" if not label.endswith("s") else label)
        elif self.lang == "hi":
            return COCO_PLURALS_HI.get(label, self.translate_label(label))
        else:  # hinglish
            return COCO_PLURALS_HINGLISH.get(label, self.translate_label(label) + "s")

    # ── Sentence builders ─────────────────────────────────────────────────

    def format_detection(
        self,
        label: str,
        direction: str,
        depth_meters: float | None,
        vertical_zone: str = "normal",
        is_approaching: bool = False,
    ) -> str:
        """
        Build a single-object detection sentence (FR-1.4, FR-3.2).

        Parameters
        ----------
        label          : COCO English label string
        direction      : ``"left"`` | ``"center"`` | ``"right"``
        depth_meters   : metric depth float, or ``None`` if depth is disabled
        vertical_zone  : ``"normal"`` | ``"overhead"`` | ``"ground"``
        is_approaching : whether object exceeds kinematic approaching threshold
        """
        l = self.translate_label(label)
        d = self.translate_direction(direction)

        if self.lang == "en":
            if vertical_zone == "overhead":
                l = f"overhead {l}"
            elif vertical_zone == "ground":
                l = f"ground-level {l}"
            if is_approaching:
                l = f"approaching {l}"
        elif self.lang == "hi":
            if vertical_zone == "overhead":
                l = f"सिर के पास {l}"
            elif vertical_zone == "ground":
                l = f"नीचे जमीन पर {l}"
            if is_approaching:
                l = f"आता हुआ {l}"
        else:  # hinglish
            if vertical_zone == "overhead":
                l = f"sir ke paas {l}"
            elif vertical_zone == "ground":
                l = f"neeche zameen par {l}"
            if is_approaching:
                l = f"aata hua {l}"

        if depth_meters is not None:
            return self._tmpl["with_depth"].format(
                label=l, dir=d, dist=f"{depth_meters:.1f}"
            )
        return self._tmpl["no_depth"].format(label=l, dir=d)

    def format_crowd_clause(
        self,
        label: str,
        direction: str,
        count: int,
        min_depth: float,
    ) -> str:
        """
        Build a compressed crowd clause (FR-3.2, FR-3.3).
        Buckets counts strictly into 2, 3, or 'multiple' (>= 4).
        """
        q_depth = quantize_distance(min_depth)
        d_str = f"{q_depth:.1f}"
        plural = self.get_plural_label(label)

        if self.lang == "en":
            if direction == "center":
                d_dir = "ahead"
            elif direction == "left":
                d_dir = "on the left"
            elif direction == "right":
                d_dir = "on the right"
            else:
                d_dir = direction

            if count in (2, 3):
                return f"{count} {plural} {d_dir}, closest at {d_str} meters"
            else:
                return f"multiple {plural} {d_dir}, closest at {d_str} meters"
        elif self.lang == "hi":
            d_dir = self.translate_direction(direction)
            if count in (2, 3):
                return f"{d_dir} {count} {plural}, सबसे पास {d_str} मीटर पर"
            else:
                return f"{d_dir} कई {plural}, सबसे पास {d_str} मीटर पर"
        else:  # hinglish
            d_dir = self.translate_direction(direction)
            if count in (2, 3):
                return f"{d_dir} {count} {plural}, sabse paas {d_str} meter par"
            else:
                return f"{d_dir} kai {plural}, sabse paas {d_str} meter par"

    def format_fast_critical(
        self,
        label: str,
        direction: str,
        depth: float,
    ) -> str:
        """
        Build a high-speed vehicle Gate C critical warning (FR-1.3, FR-3.2).
        """
        q_depth = quantize_distance(depth)
        d_str = f"{q_depth:.1f}"
        l = self.translate_label(label)

        if self.lang == "en":
            d_dir = "ahead" if direction == "center" else f"on the {direction}"
            return f"CRITICAL: Fast approaching {l} {d_dir} at {d_str} meters"
        elif self.lang == "hi":
            d_dir = self.translate_direction(direction)
            return f"CRITICAL: तेज़ आता हुआ {l} {d_dir}, {d_str} मीटर पर"
        else:  # hinglish
            d_dir = self.translate_direction(direction)
            return f"CRITICAL: Tez aata hua {l} {d_dir}, {d_str} meter par"

    def format_system_alert(
        self,
        alert_key: str,
        depth: float | None = None,
    ) -> str:
        """
        Format a system-level or curtain alert (FR-1.5, FR-2.2, FR-2.3, FR-3.2).
        Supported keys: 'obstacle_ahead', 'camera_lost', 'camera_restored', 'low_light'.
        """
        d_str = f"{quantize_distance(depth):.1f}" if depth is not None else ""
        if alert_key == "obstacle_ahead":
            if self.lang == "en":
                return f"obstacle ahead at {d_str} meters"
            elif self.lang == "hi":
                return f"सामने रुकावट, {d_str} मीटर पर"
            else:
                return f"saamne rukawat, {d_str} meter par"
        elif alert_key == "camera_lost":
            if self.lang == "en":
                return "CRITICAL: Camera signal lost"
            elif self.lang == "hi":
                return "CRITICAL: कैमरा सिग्नल बंद है"
            else:
                return "CRITICAL: Camera signal band hai"
        elif alert_key == "camera_restored":
            if self.lang == "en":
                return "Camera restored"
            elif self.lang == "hi":
                return "कैमरा चालू है"
            else:
                return "Camera chalu hai"
        elif alert_key == "low_light":
            if self.lang == "en":
                return "Camera blocked or low light"
            elif self.lang == "hi":
                return "कैमरा ढका है या अंधेरा है"
            else:
                return "Camera dhaka hai ya andhera hai"
        return self.system_message(alert_key)

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
