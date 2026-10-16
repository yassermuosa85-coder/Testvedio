# transcribe_offline.py — fallback for prep.py when huggingface.co is blocked: Whisper turbo (sherpa-onnx, from GitHub).
# usage: python transcribe_offline.py <work_dir> [--lang ar]
# Reads <work_dir>/audio16.wav (prep.py writes it before it tries Whisper), writes words.txt and transcript.json.
# The model gives no word timestamps, so the audio is split at pauses, each phrase is transcribed on its own and its
# words are spread over the phrase's voiced frames by length — good to ~0.3 s, fine for 3–5 word captions.
import sys, os, json, wave
import numpy as np, sherpa_onnx

M = os.path.expanduser("~/models/sherpa-onnx-whisper-turbo/")

def main():
    wd = sys.argv[1]; lang = sys.argv[sys.argv.index("--lang") + 1] if "--lang" in sys.argv else "ar"
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=M + "turbo-encoder.int8.onnx", decoder=M + "turbo-decoder.int8.onnx",
                                                     tokens=M + "turbo-tokens.txt", language=lang, task="transcribe", num_threads=os.cpu_count() or 4)
    w = wave.open(os.path.join(wd, "audio16.wav")); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    hop = 160; n = len(a) // hop                                        # 10 ms frames
    db = np.array([20 * np.log10(np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2)) + 1e-6) for i in range(n)])
    low = db < np.median(db) - 14
    pauses, s = [], None                                                # runs of >= 120 ms below the speech level
    for i, v in enumerate(np.append(low, False)):
        if v and s is None: s = i
        elif not v and s is not None:
            if i - s >= 12: pauses.append((s + i) // 2)
            s = None
    cuts = [0] + [p for p in pauses if 0 < p < n] + [n]
    bounds = []                                                         # split anything Whisper can't take in one go
    for c0, c1 in zip(cuts, cuts[1:]):
        while c1 - c0 > 2800:
            m = c0 + 1000 + int(np.argmin(db[c0 + 1000:c0 + 2800])); bounds.append((c0, m)); c0 = m
        bounds.append((c0, c1))
    voiced = db > np.median(db) - 12
    segs, lines = [], []
    for c0, c1 in bounds:
        if not voiced[c0:c1].any(): continue
        pad = np.zeros(4800, np.float32)                               # 0.3 s of silence each side: without it Whisper
        st = rec.create_stream()                                        # hallucinates ("اشتركوا في القناة") on clipped starts
        st.accept_waveform(16000, np.concatenate([pad, a[c0 * hop:c1 * hop], pad])); rec.decode_stream(st)
        words = st.result.text.split()
        if not words: continue
        idx = [i for i in range(c0, c1) if voiced[i]]
        L = np.array([len(x) + 1.5 for x in words]); cum = np.concatenate([[0], np.cumsum(L)]) / L.sum()
        ws = [{"w": x, "s": idx[min(len(idx) - 1, int(cum[k] * len(idx)))] / 100,
               "e": idx[min(len(idx) - 1, max(0, int(cum[k + 1] * len(idx)) - 1))] / 100} for k, x in enumerate(words)]
        segs.append({"start": ws[0]["s"], "end": ws[-1]["e"], "text": " ".join(words), "words": ws})
        lines.append(f"[{ws[0]['s']:6.2f}-{ws[-1]['e']:6.2f}] " + " ".join(f"{x['w']}@{x['s']:.2f}" for x in ws))
    json.dump(segs, open(os.path.join(wd, "transcript.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(wd, "words.txt"), "w", encoding="utf-8").write("\n".join(lines)); print("\n".join(lines))

if __name__ == "__main__":
    main()
