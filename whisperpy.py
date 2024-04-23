import whisper
import os

try:

    # clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    # cambia il titolo della finestra del terminale
    app_title = "Trascrizione con OPEN AI Whisper"
    os.system("title " + app_title)
    print(app_title.upper() + "\n")

    # ask the user for the audio file path
    audio_file = input("Inserisci il percorso del file audio da trascrivere (default: audio.mp3): ") or "audio.mp3"

    # load the model
    print("\nCaricamento modello...")
    model = whisper.load_model("base")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Linguaggio identificato nella registrazione: {max(probs, key=probs.get)}")

    # decode the audio
    print("Inizio decodifica audio. Prego attendere...")
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    print("Fine decodifica audio...")

    # print the recognized text
    print("\nINIZIO TRASCRIZIONE\n")
    print(result.text)
    print("\nFINE TRASCRIZIONE\n")

except KeyboardInterrupt:
    print("\nCtrl+C premuto. Uscita dal programma.")
except Exception as e:
    print("Programma terminat.")
    print(f"Si è verificato il seguente errore:\n{e}")