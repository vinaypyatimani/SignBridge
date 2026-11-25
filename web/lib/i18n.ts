// Interface strings. The signer side follows the sign language's written
// language; the speaker side follows the spoken language.
import type { SignLang, SpokenLang } from "./api";

export const SIGN_NAMES: Record<SignLang, string> = { ase: "ASL", gsg: "DGS", ssp: "LSE" };
export const SIGN_TO_WRITTEN: Record<SignLang, SpokenLang> = { ase: "en", gsg: "de", ssp: "es" };
export const SPOKEN_NAMES: Record<SpokenLang, string> = { en: "English", de: "Deutsch", es: "Español" };
export const BCP47: Record<SpokenLang, string> = { en: "en-US", de: "de-DE", es: "es-ES" };

const T = {
  en: {
    signer: "Signer", speaker: "Speaker",
    startCamera: "Start camera", cameraOff: "Camera is off. Start it to sign.",
    loading: "Loading hand tracking…", noModel: "No recognition model for this sign language yet.",
    spellMode: "Fingerspell", signMode: "Signs",
    youSigned: "You signed", didYouMean: "Did you mean…", send: "Send",
    holdToTalk: "Hold to talk", listening: "Listening…", typeHere: "Type a message",
    readAloud: "Read aloud", waiting: "Captions appear here when the signer finishes.",
    incoming: "Signs from the speaker appear here.",
    end: "End and summarize", summary: "Conversation summary", close: "Close",
    notInterpreter: "Communication aid, not a qualified interpreter.",
    offline: "Not connected to the server. Retrying…",
    notInLibrary: "Shown as text: no verified sign yet for",
    leftHanded: "Left-handed", flip: "Turn speaker side",
  },
  de: {
    signer: "Gebärdende Person", speaker: "Sprechende Person",
    startCamera: "Kamera starten", cameraOff: "Die Kamera ist aus. Starten Sie sie zum Gebärden.",
    loading: "Handerkennung wird geladen…", noModel: "Für diese Gebärdensprache gibt es noch kein Erkennungsmodell.",
    spellMode: "Fingeralphabet", signMode: "Gebärden",
    youSigned: "Gebärdet", didYouMean: "Meinten Sie…", send: "Senden",
    holdToTalk: "Gedrückt halten zum Sprechen", listening: "Hört zu…", typeHere: "Nachricht eingeben",
    readAloud: "Vorlesen", waiting: "Hier erscheinen Untertitel, sobald die gebärdende Person fertig ist.",
    incoming: "Hier erscheinen Gebärden der sprechenden Person.",
    end: "Beenden und zusammenfassen", summary: "Zusammenfassung", close: "Schließen",
    notInterpreter: "Kommunikationshilfe, kein qualifizierter Dolmetschdienst.",
    offline: "Keine Verbindung zum Server. Neuer Versuch…",
    notInLibrary: "Als Text gezeigt, noch keine geprüfte Gebärde für",
    leftHanded: "Linkshändig", flip: "Sprecherseite drehen",
  },
  es: {
    signer: "Persona signante", speaker: "Persona oyente",
    startCamera: "Iniciar cámara", cameraOff: "La cámara está apagada. Iníciela para signar.",
    loading: "Cargando el seguimiento de manos…", noModel: "Aún no hay modelo de reconocimiento para esta lengua de signos.",
    spellMode: "Dactilología", signMode: "Signos",
    youSigned: "Ha signado", didYouMean: "¿Quiso decir…?", send: "Enviar",
    holdToTalk: "Mantenga pulsado para hablar", listening: "Escuchando…", typeHere: "Escriba un mensaje",
    readAloud: "Leer en voz alta", waiting: "Aquí aparecen los subtítulos cuando la persona signante termina.",
    incoming: "Aquí aparecen los signos de la persona oyente.",
    end: "Terminar y resumir", summary: "Resumen de la conversación", close: "Cerrar",
    notInterpreter: "Ayuda a la comunicación, no un intérprete cualificado.",
    offline: "Sin conexión con el servidor. Reintentando…",
    notInLibrary: "Se muestra como texto, aún sin signo verificado para",
    leftHanded: "Zurdo", flip: "Girar lado oyente",
  },
} as const;

export type Strings = Record<keyof (typeof T)["en"], string>;
export const strings = (lang: SpokenLang): Strings => T[lang];
