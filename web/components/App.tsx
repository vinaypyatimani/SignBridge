"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Conversation, fetchModes, transcribe, type Caption, type ServerMessage, type SignLang,
  type SignSequence, type SpokenLang } from "@/lib/api";
import { BCP47, SIGN_NAMES, SIGN_TO_WRITTEN, SPOKEN_NAMES, strings } from "@/lib/i18n";
import type { Candidate } from "@/lib/recognizer";
import SignerCamera from "./SignerCamera";
import SignPlayer from "./SignPlayer";

type Turn = { id: number; who: "signer" | "speaker"; text: string; detail?: string };

export default function App() {
  const [signLang, setSignLang] = useState<SignLang>("ase");
  const [spokenLang, setSpokenLang] = useState<SpokenLang>("en");
  const [mode, setMode] = useState<string>("");
  const [modes, setModes] = useState<{ id: string; title: Record<SpokenLang, string> }[]>([]);
  const [leftHanded, setLeftHanded] = useState(false);
  const [spelling, setSpelling] = useState(false);
  const [flipped, setFlipped] = useState(false);
  const [readAloud, setReadAloud] = useState(true);
  const [online, setOnline] = useState(false);
  const [pending, setPending] = useState<Candidate[][]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [confirm, setConfirm] = useState<Caption | null>(null);
  const [sequence, setSequence] = useState<SignSequence | null>(null);
  const [draft, setDraft] = useState("");
  const [recording, setRecording] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [summary, setSummary] = useState<string | null>(null);

  const conv = useRef<Conversation | null>(null);
  const lastSent = useRef<Candidate[][]>([]);
  const recorder = useRef<MediaRecorder | null>(null);
  const settings = { signLang, spokenLang, mode: mode || null };
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const tSigner = strings(SIGN_TO_WRITTEN[signLang]);
  const tSpeaker = strings(spokenLang);

  const addTurn = useCallback((who: Turn["who"], text: string, detail?: string) => {
    setTurns((ts) => [...ts, { id: Date.now() + Math.random(), who, text, detail }]);
  }, []);

  const readAloudRef = useRef(readAloud);
  readAloudRef.current = readAloud;
  const speak = useCallback((text: string) => {
    if (!readAloudRef.current || !("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = BCP47[settingsRef.current.spokenLang];
    speechSynthesis.speak(u);
  }, []);

  const onMessage = useCallback((m: ServerMessage) => {
    if (m.type === "caption") {
      if (m.confirm) { setConfirm(m); return; }
      setConfirm(null);
      addTurn("signer", m.text, m.glosses.join(" "));
      speak(m.text);
    } else if (m.type === "sign_sequence") {
      setSequence(m);
    } else {
      setNotice(m.message);
    }
  }, [addTurn, speak]);

  useEffect(() => {
    conv.current = new Conversation(onMessage, (s) => setOnline(s === "open"));
    fetchModes().then(setModes).catch(() => setModes([]));
    return () => conv.current?.close();
  }, [onMessage]);

  const flush = useCallback(() => {
    setPending((p) => {
      if (p.length) {
        lastSent.current = p;
        conv.current?.sendSigns(p, settingsRef.current);
      }
      return [];
    });
  }, []);

  function chooseOption(index: number, gloss: string) {
    const fixed = lastSent.current.map((c, i) => (i === index ? [[gloss, 1] as Candidate] : c));
    lastSent.current = fixed;
    conv.current?.sendSigns(fixed, settings);
  }

  function acceptAlternative(text: string) {
    setConfirm(null);
    addTurn("signer", text);
    speak(text);
  }

  function sendDraft() {
    const text = draft.trim();
    if (!text) return;
    addTurn("speaker", text);
    conv.current?.sendText(text, settings);
    setDraft("");
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      rec.ondataavailable = (e) => chunks.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        try {
          const text = await transcribe(new Blob(chunks, { type: rec.mimeType }), spokenLang);
          if (text) { addTurn("speaker", text); conv.current?.sendText(text, settingsRef.current); }
        } catch (e) {
          setNotice(e instanceof Error ? e.message : String(e));
        }
      };
      rec.start();
      recorder.current = rec;
      setRecording(true);
    } catch (e) {
      setNotice(e instanceof Error ? e.message : String(e));
    }
  }

  function stopRecording() {
    recorder.current?.stop();
    recorder.current = null;
    setRecording(false);
  }

  async function endConversation() {
    try {
      const r = await conv.current!.summary(SIGN_TO_WRITTEN[signLang]);
      setSummary(r.summary);
    } catch (e) {
      setNotice(e instanceof Error ? e.message : String(e));
    }
  }

  const lastSignerTurn = [...turns].reverse().find((t) => t.who === "signer");

  return (
    <div className="app">
      <header className="bar">
        <h1 className="brand">TriSign</h1>
        <label>Sign language
          <select value={signLang} onChange={(e) => { setSignLang(e.target.value as SignLang); setPending([]); }}>
            {(Object.keys(SIGN_NAMES) as SignLang[]).map((l) => <option key={l} value={l}>{SIGN_NAMES[l]}</option>)}
          </select>
        </label>
        <label>Spoken language
          <select value={spokenLang} onChange={(e) => setSpokenLang(e.target.value as SpokenLang)}>
            {(Object.keys(SPOKEN_NAMES) as SpokenLang[]).map((l) => <option key={l} value={l}>{SPOKEN_NAMES[l]}</option>)}
          </select>
        </label>
        <label>Setting
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="">General</option>
            {modes.map((m) => <option key={m.id} value={m.id}>{m.title[spokenLang] ?? m.id}</option>)}
          </select>
        </label>
        <p className="disclaimer">{tSpeaker.notInterpreter}</p>
      </header>

      {!online && <p className="banner" role="status">{tSpeaker.offline}</p>}
      {notice && <p className="banner attention" role="alert" onClick={() => setNotice(null)}>{notice}</p>}

      <main className="sides">
        <section className="side side-signer" aria-label={tSigner.signer}>
          <div className="side-head">
            <h2>{tSigner.signer} <span className="lang">{SIGN_NAMES[signLang]}</span></h2>
            <div className="toggles">
              <button className={`chip ${!spelling ? "on" : ""}`} aria-pressed={!spelling} onClick={() => setSpelling(false)}>{tSigner.signMode}</button>
              <button className={`chip ${spelling ? "on" : ""}`} aria-pressed={spelling} onClick={() => setSpelling(true)}>{tSigner.spellMode}</button>
              <label className="check"><input type="checkbox" checked={leftHanded} onChange={(e) => setLeftHanded(e.target.checked)} /> {tSigner.leftHanded}</label>
            </div>
          </div>

          <SignerCamera signLang={signLang} leftHanded={leftHanded} spelling={spelling} t={tSigner}
            onSign={(c) => setPending((p) => [...p, c])}
            onSpelled={(w) => setPending((p) => [...p, [[`FS:${w}`, 1]]])}
            onUtteranceEnd={flush} />

          {pending.length > 0 && (
            <div className="pending">
              <span className="muted">{tSigner.youSigned}:</span>
              <span className="gloss-line">{pending.map((c) => c[0][0]).join(" ")}</span>
              <button className="button signer small" onClick={flush}>{tSigner.send}</button>
            </div>
          )}

          {confirm?.confirm && (
            <div className="confirm" role="alertdialog" aria-label={tSigner.didYouMean}>
              <p className="confirm-title">{tSigner.didYouMean}</p>
              {confirm.confirm.uncertain_signs.map((u) => (
                <div key={u.index} className="confirm-row">
                  {u.options.map((g) => <button key={g} className="chip" onClick={() => chooseOption(u.index, g)}>{g}</button>)}
                </div>
              ))}
              {[confirm.text, ...confirm.confirm.alternatives].filter(Boolean).map((a) => (
                <button key={a} className="confirm-sentence" onClick={() => acceptAlternative(a)}>{a}</button>
              ))}
            </div>
          )}

          <SignPlayer sequence={sequence} speed={1} emptyText={tSigner.incoming} />
          {sequence && sequence.not_in_library.length > 0 && (
            <p className="muted small">{tSigner.notInLibrary} {sequence.not_in_library.join(", ")}.</p>
          )}
        </section>

        <section className={`side side-speaker ${flipped ? "flipped" : ""}`} aria-label={tSpeaker.speaker}>
          <div className="side-head">
            <h2>{tSpeaker.speaker} <span className="lang">{SPOKEN_NAMES[spokenLang]}</span></h2>
            <div className="toggles">
              <label className="check"><input type="checkbox" checked={readAloud} onChange={(e) => setReadAloud(e.target.checked)} /> {tSpeaker.readAloud}</label>
              <button className="chip" aria-pressed={flipped} onClick={() => setFlipped((f) => !f)}>{tSpeaker.flip}</button>
            </div>
          </div>

          <p className={`caption ${lastSignerTurn ? "" : "muted"}`} aria-live="polite">
            {lastSignerTurn?.text ?? tSpeaker.waiting}
          </p>

          <ol className="transcript">
            {turns.map((t) => (
              <li key={t.id} className={`turn ${t.who}`}>
                <span className="turn-text">{t.text}</span>
                {t.detail && <span className="turn-detail">{t.detail}</span>}
              </li>
            ))}
          </ol>

          <div className="compose">
            <button className={`button speaker mic ${recording ? "recording" : ""}`}
              onPointerDown={startRecording} onPointerUp={stopRecording} onPointerLeave={() => recording && stopRecording()}>
              {recording ? tSpeaker.listening : tSpeaker.holdToTalk}
            </button>
            <form className="type" onSubmit={(e) => { e.preventDefault(); sendDraft(); }}>
              <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder={tSpeaker.typeHere} lang={spokenLang} aria-label={tSpeaker.typeHere} />
              <button className="button speaker" type="submit">{tSpeaker.send}</button>
            </form>
          </div>
          <button className="link end" onClick={endConversation}>{tSpeaker.end}</button>
        </section>
      </main>

      {summary !== null && (
        <dialog open className="summary">
          <h2>{tSigner.summary}</h2>
          <pre>{summary}</pre>
          <button className="button signer" onClick={() => setSummary(null)}>{tSigner.close}</button>
        </dialog>
      )}
    </div>
  );
}
