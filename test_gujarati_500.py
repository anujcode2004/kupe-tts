#!/usr/bin/env python3
"""OmniVoice Gujarati streaming test (~500-word sample).

Streams a long-form Gujarati passage through the realtime WebSocket player at
``/ws/tts``, plays the first chunk as soon as it arrives, and continues
playing subsequent chunks back-to-back.

Usage
-----
  # Default voice = shanti, language = gu, speed = 1.0 (server default)
  python test_gujarati_500.py

  # Different voice / speed
  python test_gujarati_500.py --voice ajay --speed 1.2
  python test_gujarati_500.py --voice shanti --speed 0.85

  # Save first / rest / full WAV files into ./streaming_output/
  python test_gujarati_500.py --save

  # Run against a remote server
  python test_gujarati_500.py --url ws://192.168.1.10:8000/ws/tts

Notes
-----
* The corpus contains ~500 words of authentic Gujarati prose with proper full
  stops (``.``) and clause-level punctuation (``,``) so OmniVoice's chunk
  splitter can produce natural-sounding paragraphs.
* Speed range: 0.25 ≤ speed ≤ 3.0  (validated client-side and server-side).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Reuse the realtime player implementation.  Keeping the import lazy here so
# we surface a friendly error if the user runs this from a different cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from test_ws_realtime_player import (  # noqa: E402  (sibling script)
        OUTPUT_DIR,
        _BOLD,
        _DIM,
        _RED,
        _c,
        stream_and_play,
    )
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"Could not import test_ws_realtime_player: {exc}\n"
        "Run this script from the OmniVoice root directory."
    )


# ---------------------------------------------------------------------------
# 500-word Gujarati corpus — authentic prose, proper full stops & punctuation.
# ---------------------------------------------------------------------------
GUJARATI_TEXT = (

    "ਕ੍ਰਿਸ਼ੀ ਪੰਜਾਬ ਦੀ ਆਤਮਾ ਹੈ। ਜਦੋਂ ਪਹਿਲਾ ਮੀਂਹ ਪੈਂਦਾ ਹੈ, ਤਾਂ ਧਰਤੀ ਆਪਣੀ ਤਰਸ ਬੁਝਾ ਕੇ ਹਰੀ-ਭਰੀ ਹੋ ਜਾਂਦੀ ਹੈ ਅਤੇ ਹਵਾ ਵਿੱਚ ਮਿੱਟੀ ਦੀ ਮਿੱਠੀ ਖੁਸ਼ਬੂ ਫੈਲ ਜਾਂਦੀ ਹੈ। ਕਿਸਾਨ ਕਪਾਹ, ਮੂੰਗਫਲੀ, ਗੰਧਮ, ਬਾਜਰਾ ਅਤੇ ਜਵਾਰ ਵਰਗੀਆਂ ਕਈ ਫਸਲਾਂ ਉਗਾਉਂਦੇ ਹਨ। ਹਰ ਇਕ ਫਸਲ ਪਿੱਛੇ ਸਾਲਾਂ ਦੀ ਮਿਹਨਤ, ਗਹਿਰੀ ਅਾਸਥਾ ਅਤੇ ਪਰਿਵਾਰ ਦਾ ਮਿਲ ਜੁਲ ਕੇ ਕੀਤਾ ਸੰਗਠਿਤ ਪੈਣਪ ਕਰ ਸਕਦਾ ਹੈ। ਜਦੋਂ ਫਸਲ ਲਣ ਲਈ ਤਿਆਰ ਹੁੰਦੀ ਹੈ, ਤਾਂ ਸਭ ਤੋਂ ਪਹਿਲਾਂ ਉਹ ਮੰਦਰ ਵਿੱਚ ਭੇਟ ਕੀਤੀ ਜਾਂਦੀ ਹੈ ਅਤੇ ਫਿਰ ਹੀ ਅਨਾਜ ਘਰ ਦੀ ਤਿਜ਼ੌਰੀ ਵਿੱਚ ਰੱਖਿਆ ਜਾਂਦਾ ਹੈ। \n\n"

    "नमस्ते friends! पेश है प्राकृतिक मिठास का राजा — सीताफल, એટલે કે delicious Custard Apple! इसकी creamy texture, शानदार aroma और naturally sweet taste हर bite को खास बना देती है. જ્યારે તમે એકવાર સીતાફળનો સ્વાદ માણશો, ત્યારે તેની મીઠાશ તમને વારંવાર યાદ આવશે. Vitamins, fiber और natural energy से भरपूर सीताफल सिर्फ एक fruit नहीं, बल्कि पूरे family के लिए healthy lifestyle choice है. Kids हो, adults हो या elders — सबको इसका soft pulp और refreshing taste बहुत पसंद आता है.  आज की busy life में हर कोई चाहता है कुछ tasty भी हो और healthy भी, और सीताफल दोनों का perfect combination है. यह energy boost करने में मदद करता है, digestion को support करता है और naturally fresh feeling देता है. ગુજરાતના તાજા ખેતરોમાંથી સીધું તમારા ઘર સુધી પહોંચતું આ fruit purity અને quality નો સાચો અનુભવ કરાવે છે.  अगर आप ऐसा fruit ढूंढ रहे हैं जो premium, nutritious और authentic natural taste से भरपूर हो, तो सीताफल आपके लिए best choice है. Milkshakes, ice creams, desserts या smoothies — हर चीज़ में इसका flavor extra special बना देता है. તો હવે રાહ શા માટે? ઘરે લાવો કુદરતની મીઠાશ, માણો રાજાશાહી સ્વાદવાળું સીતાફળ અને દરેક bite સાથે health, happiness અને amazing taste નો આનંદ લો!"

    "गुजरात के त्यौहार यहाँ के सामूहिक जीवन का हृदय हैं। नवरात्रि के नौ दिनों तक "
    "पूरा राज्य गरबा और डांडिया की ताल पर झूमता रहता है। युवक और युवतियाँ रंग-बिरंगे "
    "परिधान पहनकर माता की आरती में शामिल होते हैं, और रात भर ढोल-नगाड़ों की गूंज के साथ "
    "भक्ति और उल्लास का संगम महसूस होता है। उत्तरायण के दिन आकाश रंगीन पतंगों से "
    "भर जाता है, और बच्चों की 'काट्यो छे' की पुकार हर घर में गूंजती है। दिवाली पर "
    "हर घर के आँगन में रंगोली बनती है, और छोटे-छोटे दीयों की माला से गलियाँ चमक उठती हैं। \n\n"


    "ગુજરાતી ભોજન પણ આપણા ગૌરવનો અતૂટ હિસ્સો છે. ઢોકળાં, ખાંડવી, થેપલાં, "
    "ફાફડા, જલેબી અને ઊંધિયું જેવી વાનગીઓ માત્ર સ્વાદની નહીં, પણ આત્મીયતાની "
    "નિશાની છે. ઘરમાં મહેમાન આવે ત્યારે પ્રથમ પાણી, પછી ગોળ-ઘી અને છેવટે "
    "ગરમ ચા આપવી — એ આપણી પરંપરા છે. કોઈ પણ વ્યક્તિ ભૂખ્યો આપણા ઘરેથી "
    "પાછો ન જાય, એ સંકલ્પ ગુજરાતી પરિવારોના હૃદયમાં વસ્યો છે. દરેક રાંધણ "
    "પાછળ માતાનો પ્રેમ, દાદીના આશીર્વાદ અને પેઢીઓની યાદો સમાયેલી છે. એટલે જ "
    "આપણા રસોડામાંથી નીકળતી સુગંધ માત્ર ભૂખ નહીં, પણ આત્માની તૃપ્તિ આપે છે."


    "গ্রামজীবনের প্রধান আকর্ষণ এখানকার সরলতা এবং নিষ্কপট স্নেহ। সন্ধ্যেবেলায় "
    "বয়োজ্যেষ্ঠেরা বারান্দায় বসে শিশুদের রামায়ণ, মহাভারত এবং নরসিংহ মেহেতার "
    "ভক্তিগাথা শোনান। মায়েরা চুলায় গরম রুটি, সবজি ও খিচুড়ি তৈরি করেন, "
    "আর ঘরের পরিবেশ ঘি ও মশলার সুবাসে ভরে ওঠে। গ্রামের স্কুলে "
    "শিক্ষকরা শুধু পুস্তকজ্ঞানই নয়, বরং সংস্কার, অহিংসা ও সত্যের গুরুত্বও "
    "শিখিয়ে দেন। প্রতিটি পাঠের সাথে গান্ধীজি ও সর্দার প্যাটেলের আদর্শ শিশুদের মনে "
    "গভীরে গেঁথে যায়। \n\n"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniVoice Gujarati streaming test (~500-word sample).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="ws://localhost:8000/ws/tts",
                        help="WebSocket endpoint of the OmniVoice server.")
    parser.add_argument("--voice", "-v", default="shanti", metavar="NAME",
                        help="Voice profile (must be preloaded server-side).")
    parser.add_argument("--language", "-l", default="gu", metavar="CODE",
                        help="Language code. 'gu' = Gujarati. 'auto' for auto-detect.")
    parser.add_argument("--speed", "-s", type=float, default=None, metavar="X",
                        help=("Speaking-speed multiplier (0.25–3.0). "
                              "1.0 = normal, <1.0 slower, >1.0 faster. "
                              "Omit for server default."))
    parser.add_argument("--no-play", dest="play", action="store_false", default=True,
                        help="Disable realtime audio playback (still saves WAVs).")
    parser.add_argument("--save", action="store_true", default=False,
                        help="Save first/rest/full WAV files under ./streaming_output/.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--recv-timeout", type=float, default=180.0,
                        help="Per-message receive timeout (seconds).")
    parser.add_argument("--print-text", action="store_true",
                        help="Print the full Gujarati corpus and exit.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    text = GUJARATI_TEXT.strip()
    word_count = len(text.split())
    char_count = len(text)

    if args.print_text:
        print(text)
        sys.exit(0)

    speed = args.speed
    if speed is not None and not (0.25 <= speed <= 3.0):
        sys.exit(f"--speed must be between 0.25 and 3.0 (got {speed}).")

    language = None if args.language.lower() in ("auto", "none", "") else args.language
    voice    = args.voice if args.voice else None

    print(_c(_BOLD, "\n╔══ OmniVoice Gujarati 500-word Streaming Test ══╗"))
    print(f"  url        : {args.url}")
    print(f"  voice      : {voice or '(server default)'}")
    print(f"  language   : {language or 'auto (server default)'}")
    print(f"  speed      : {speed if speed is not None else '(server default)'}")
    print(f"  playback   : {args.play}")
    print(f"  save wavs  : {args.save}")
    print(f"  word count : ~{word_count}")
    print(f"  char count : {char_count}")
    print(_c(_BOLD, "╚════════════════════════════════════════════════╝\n"))
    print(_c(_DIM, text[:200] + ("..." if len(text) > 200 else "")), flush=True)
    print()

    try:
        asyncio.run(
            stream_and_play(
                url=args.url,
                text=text,
                play=args.play,
                save=args.save,
                output_dir=args.output_dir,
                recv_timeout=args.recv_timeout,
                language=language,
                voice=voice,
                speed=speed,
            )
        )
    except KeyboardInterrupt:
        print(_c(_DIM, "\nInterrupted."))
    except Exception as exc:
        sys.exit(_c(_RED, f"ERROR: {exc}"))


if __name__ == "__main__":
    main()
