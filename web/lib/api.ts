// Backend client: one WebSocket per conversation plus a few REST calls.
import type { Candidate } from "./recognizer";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type SignLang = "ase" | "gsg" | "ssp";
export type SpokenLang = "en" | "de" | "es";

export type Caption = {
  type: "caption";
  text: string;
  glosses: string[];
  confidence: number;
  confirm: null | { uncertain_signs: { index: number; options: string[] }[]; alternatives: string[] };
};
export type SequenceItem =
  | { type: "sign"; gloss: string; frames: number[][][] }
  | { type: "spell"; gloss: string; letters: { letter: string; frames: number[][][] | null }[] }
  | { type: "text"; gloss: string; text: string };
export type SignSequence = {
  type: "sign_sequence"; text: string; simplified: string; glosses: string[];
  items: SequenceItem[]; fps: number; not_in_library: string[];
};
export type ServerMessage = Caption | SignSequence | { type: "error"; message: string };

export type Settings = { signLang: SignLang; spokenLang: SpokenLang; mode: string | null };

export class Conversation {
  private ws: WebSocket | null = null;
  private queue: string[] = [];
  private closed = false;
  readonly id = crypto.randomUUID();

  constructor(private onMessage: (m: ServerMessage) => void, private onStatus: (s: "open" | "closed") => void) {
    this.connect();
  }

  private connect() {
    const ws = new WebSocket(`${API_URL.replace(/^http/, "ws")}/ws/${this.id}`);
    ws.onopen = () => {
      this.onStatus("open");
      for (const m of this.queue.splice(0)) ws.send(m);
    };
    ws.onmessage = (e) => this.onMessage(JSON.parse(e.data));
    ws.onclose = () => {
      this.onStatus("closed");
      if (!this.closed) setTimeout(() => this.connect(), 1500);
    };
    this.ws = ws;
  }

  private send(msg: object) {
    const s = JSON.stringify(msg);
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(s);
    else this.queue.push(s);
  }

  sendSigns(signs: Candidate[][], s: Settings) {
    this.send({ type: "signs", sign_lang: s.signLang, spoken_lang: s.spokenLang, mode: s.mode,
                signs: signs.map((candidates) => ({ candidates })) });
  }

  sendText(text: string, s: Settings) {
    this.send({ type: "text", text, sign_lang: s.signLang, spoken_lang: s.spokenLang, mode: s.mode });
  }

  async summary(lang: SpokenLang): Promise<{ summary: string; turns: string[] }> {
    const r = await fetch(`${API_URL}/api/sessions/${this.id}/summary`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ lang }),
    });
    if (!r.ok) throw new Error(`Summary failed (${r.status})`);
    return r.json();
  }

  close() {
    this.closed = true;
    this.ws?.close();
  }
}

export async function fetchModes(): Promise<{ id: string; title: Record<SpokenLang, string> }[]> {
  const r = await fetch(`${API_URL}/api/modes`);
  return r.ok ? r.json() : [];
}

export async function transcribe(audio: Blob, lang: SpokenLang): Promise<string> {
  const body = new FormData();
  body.append("audio", audio, "speech.webm");
  body.append("lang", lang);
  const r = await fetch(`${API_URL}/api/transcribe`, { method: "POST", body });
  if (!r.ok) throw new Error(r.status === 501 ? "Speech input is not installed on the server." : `Transcription failed (${r.status})`);
  return (await r.json()).text as string;
}
