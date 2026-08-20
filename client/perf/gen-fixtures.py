#!/usr/bin/env python3
import json, os, struct, time, urllib.request
from pathlib import Path

OUT = Path("/tmp/sp-fixtures"); OUT.mkdir(exist_ok=True)
TARGET = 25.0
URL = os.environ.get("KOKORO_URL", "http://127.0.0.1:18800/v1/audio/speech")
VOICE = os.environ.get("KOKORO_VOICE", "af_heart")

PASSAGES = [
    "It was the best of times, it was the worst of times, it was the age "
    "of wisdom, it was the age of foolishness, it was the epoch of belief, "
    "it was the epoch of incredulity, it was the season of light, it was "
    "the season of darkness, it was the spring of hope, it was the winter "
    "of despair. We had everything before us, we had nothing before us.",
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife. However little known the "
    "feelings or views of such a man may be on his first entering a "
    "neighbourhood, this truth is so well fixed in the minds of the "
    "surrounding families, that he is considered as the rightful property "
    "of some one or other of their daughters.",
    "Call me Ishmael. Some years ago, never mind how long precisely, having "
    "little or no money in my purse, and nothing particular to interest me "
    "on shore, I thought I would sail about a little and see the watery "
    "part of the world. It is a way I have of driving off the spleen and "
    "regulating the circulation. Whenever I find myself growing grim, I "
    "account it high time to get to sea as soon as I can.",
    "To be, or not to be, that is the question. Whether tis nobler in the "
    "mind to suffer the slings and arrows of outrageous fortune, or to take "
    "arms against a sea of troubles, and by opposing end them. To die, to "
    "sleep, no more, and by a sleep to say we end the heart-ache and the "
    "thousand natural shocks that flesh is heir to. Tis a consummation "
    "devoutly to be wished.",
    "Shall I compare thee to a summers day? Thou art more lovely and more "
    "temperate. Rough winds do shake the darling buds of May, and summers "
    "lease hath all too short a date. Sometime too hot the eye of heaven "
    "shines, and often is his gold complexion dimmed. And every fair from "
    "fair sometime declines, by chance, or natures changing course untrimmed. "
    "But thy eternal summer shall not fade.",
    "Four score and seven years ago our fathers brought forth on this "
    "continent, a new nation, conceived in liberty, and dedicated to the "
    "proposition that all men are created equal. Now we are engaged in a "
    "great civil war, testing whether that nation, or any nation so "
    "conceived and so dedicated, can long endure. We are met on a great "
    "battle-field of that war.",
    "When in the course of human events, it becomes necessary for one "
    "people to dissolve the political bands which have connected them with "
    "another, and to assume among the powers of the earth, the separate "
    "and equal station to which the laws of nature and of natures God "
    "entitle them, a decent respect to the opinions of mankind requires "
    "that they should declare the causes which impel them to the separation.",
    "In the beginning God created the heaven and the earth. And the earth "
    "was without form, and void; and darkness was upon the face of the "
    "deep. And the Spirit of God moved upon the face of the waters. And "
    "God said, Let there be light, and there was light. And God saw the "
    "light, that it was good, and God divided the light from the darkness.",
    "Once upon a midnight dreary, while I pondered, weak and weary, over "
    "many a quaint and curious volume of forgotten lore, while I nodded, "
    "nearly napping, suddenly there came a tapping, as of some one gently "
    "rapping, rapping at my chamber door. Tis some visitor, I muttered, "
    "tapping at my chamber door. Only this and nothing more. Ah, "
    "distinctly I remember it was in the bleak December.",
    "Two roads diverged in a yellow wood, and sorry I could not travel "
    "both, and be one traveler, long I stood, and looked down one as far "
    "as I could, to where it bent in the undergrowth. Then took the other, "
    "as just as fair, and having perhaps the better claim, because it was "
    "grassy and wanted wear, though as for that the passing there had "
    "worn them really about the same.",
    "In the year 1878 I took my degree of Doctor of Medicine of the "
    "University of London, and proceeded to Netley to go through the "
    "course prescribed for surgeons in the army. Having completed my "
    "studies there, I was duly attached to the Fifth Northumberland "
    "Fusiliers as Assistant Surgeon. The regiment was stationed in "
    "India at the time, and before I could join it, the second Afghan "
    "war had broken out.",
    "Alice was beginning to get very tired of sitting by her sister on "
    "the bank, and of having nothing to do. Once or twice she had peeped "
    "into the book her sister was reading, but it had no pictures or "
    "conversations in it, and what is the use of a book, thought Alice, "
    "without pictures or conversations? So she was considering whether "
    "the pleasure of making a daisy chain would be worth the trouble of "
    "getting up and picking the daisies.",
    "Marley was dead, to begin with. There is no doubt whatever about "
    "that. The register of his burial was signed by the clergyman, the "
    "clerk, the undertaker, and the chief mourner. Scrooge signed it, "
    "and Scrooges name was good upon Change for anything he chose to put "
    "his hand to. Old Marley was as dead as a door-nail. Mind, I dont "
    "mean to say that I know, of my own knowledge, what there is "
    "particularly dead about a door-nail.",
    "You will rejoice to hear that no disaster has accompanied the "
    "commencement of an enterprise which you have regarded with such evil "
    "forebodings. I arrived here yesterday, and my first task is to "
    "assure my dear sister of my welfare and increasing confidence in the "
    "success of my undertaking. I am already far north of London, and as "
    "I walk in the streets of Petersburgh, I feel a cold northern breeze "
    "play upon my cheeks.",
    "Left Munich at eight thirty-five p.m., on first May, arriving at "
    "Vienna early next morning; should have arrived at six forty-six, but "
    "train was an hour late. Buda-Pesth seems a wonderful place, from the "
    "glimpse which I got of it from the train and the little I could "
    "walk through the streets. I feared to go very far from the station, "
    "as we had arrived late and would start as near the correct time as "
    "possible.",
    "Squire Trelawney, Doctor Livesey, and the rest of these gentlemen "
    "having asked me to write down the whole particulars about Treasure "
    "Island, from the beginning to the end, keeping nothing back but the "
    "bearings of the island, and that only because there is still "
    "treasure not yet lifted, I take up my pen in the year of grace "
    "seventeen hundred, and go back to the time when my father kept the "
    "Admiral Benbow inn.",
    "Dorothy lived in the midst of the great Kansas prairies, with Uncle "
    "Henry, who was a farmer, and Aunt Em, who was the farmers wife. "
    "Their house was small, for the lumber to build it had to be carried "
    "by wagon many miles. There were four walls, a floor and a roof, "
    "which made one room; and this room contained a rusty looking "
    "cookstove, a cupboard for the dishes, a table, three or four chairs, "
    "and the beds.",
    "The year 1866 was signalised by a remarkable incident, a mysterious "
    "and puzzling phenomenon, which doubtless no one has yet forgotten. "
    "Not to mention rumours which agitated the maritime population and "
    "excited the public mind, even in the interior of continents, "
    "seafaring men were particularly excited. Merchants and sailors, "
    "captains of vessels, officers of naval marine, all were interested.",
    "The Time Traveller, for so it will be convenient to speak of him, "
    "was expounding a recondite matter to us. His grey eyes shone and "
    "twinkled, and his usually pale face was flushed and animated. The "
    "fire burned brightly, and the soft radiance of the incandescent "
    "lights in the lilies of silver caught the bubbles that flashed and "
    "passed in our glasses. Our chairs, being his patents, embraced and "
    "caressed us rather than submitted to be sat upon.",
    "It is very seldom that mere ordinary people like John and myself "
    "secure ancestral halls for the summer. A colonial mansion, a "
    "hereditary estate, I would say a haunted house, and reach the height "
    "of romantic felicity, but that would be asking too much of fate. "
    "Still I will proudly declare that there is something queer about it. "
    "Else, why should it be let so cheaply? And why have stood so long "
    "untenanted?",
    "A hare one day ridiculed the short feet and slow pace of the "
    "tortoise, who replied, laughing: Though you be swift as the wind, I "
    "will beat you in a race. The hare, believing her assertion to be "
    "simply impossible, assented to the proposal; and they agreed that "
    "the fox should choose the course and fix the goal. On the day "
    "appointed for the race, the two started together. The tortoise "
    "never for a moment stopped.",
]
assert len(PASSAGES) == 21

def tts(text):
    body = json.dumps({"input": text, "model": "kokoro", "voice": VOICE,
                       "response_format": "wav"}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": "Bearer dummy", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def parse(buf):
    pos = 12
    fmt = doff = dlen = None
    while pos + 8 <= len(buf):
        cid, csize = buf[pos:pos+4], struct.unpack("<I", buf[pos+4:pos+8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", buf[pos+8:pos+24])
        elif cid == b"data":
            doff = pos + 8
            dlen = csize if csize != 0xFFFFFFFF else len(buf) - doff
            break
        pos += 8 + csize
    _, ch, sr, _, _, bits = fmt
    return sr, ch, bits, doff, dlen

def trim(buf, sec):
    sr, ch, bits, doff, dlen = parse(buf)
    bps = (bits // 8) * ch
    target = int(round(sec * sr)) * bps
    pcm = buf[doff:doff + dlen]
    pcm = pcm[:target] if len(pcm) >= target else pcm + b"\x00" * (target - len(pcm))
    out = bytearray(buf[:doff])
    out[4:8] = struct.pack("<I", (doff - 8) + len(pcm))
    out[doff - 4:doff] = struct.pack("<I", len(pcm))
    out.extend(pcm)
    return bytes(out)

for i, text in enumerate(PASSAGES, 1):
    out = OUT / f"p{i:02d}.wav"
    if out.exists():
        print(f"  p{i:02d}.wav: skip"); continue
    t0 = time.monotonic()
    out.write_bytes(trim(tts(text), TARGET))
    print(f"  p{i:02d}.wav: {time.monotonic() - t0:.1f}s")
print(f"21 fixtures in {OUT}")
