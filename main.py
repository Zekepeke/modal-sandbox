import os
import re
import modal
import tempfile
import subprocess

model = modal.Cls.from_name("example-inference", "Model")

ELEVENLABS_API_KEY = "ADD API KEY HERE"


def record_audio(duration=5, sample_rate=16000) -> bytes:
    """Record audio from the microphone and return raw WAV bytes."""
    import io
    import sounddevice as sd
    import soundfile as sf

    print(f"🎙️  Listening for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("✅ Recording complete. Sending to Modal for transcription...")

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def clean_response(text: str) -> str:
    """Remove <think>...</think> tags and emojis from model output."""
    # Remove <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove emojis and special unicode characters
    text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)
    return text.strip()


def speak_text(text: str):
    """Speak text aloud using ElevenLabs TTS with macOS afplay."""
    from elevenlabs.client import ElevenLabs

    if not ELEVENLABS_API_KEY:
        print("⚠️  ELEVENLABS_API_KEY not set. Skipping TTS.")
        return

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    # Clean the text before sending to TTS
    clean_text = clean_response(text)
    if not clean_text:
        print("⚠️  Nothing to speak after cleaning response.")
        return

    print("🔊 Generating speech...")
    audio_generator = client.text_to_speech.convert(
        text=clean_text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # "George"
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    # Collect all audio chunks from the generator
    audio_bytes = b"".join(audio_generator)

    # Write to temp file and play with macOS afplay
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        print("🔊 Playing response...")
        subprocess.run(["afplay", temp_path], check=True)
    finally:
        os.unlink(temp_path)


def voice_dialogue(m, duration=5):
    """Continuous voice-based dialogue loop: STT → LLM → TTS (ElevenLabs)."""
    print("\n🎤 Voice Dialogue Mode (ElevenLabs TTS)")
    print("   Speak after the prompt. Say 'quit' to exit this mode.")
    print(f"   Recording {duration}s per turn.\n")

    m.clear_history.remote()

    while True:
        try:
            audio_bytes = record_audio(duration=duration)
            user_text = m.transcribe.remote(audio_bytes)

            if not user_text:
                print("❌ Could not transcribe. Try again.\n")
                continue

            print(f"🗣️  You said: {user_text}")

            if user_text.strip().lower() in ("quit", "quit.", "exit", "stop"):
                print("👋 Exiting voice dialogue mode.\n")
                speak_text("Goodbye!")
                m.clear_history.remote()
                break

            response = m.dialogue.remote(user_text)
            cleaned = clean_response(response)
            print(f"\n🤖 Assistant: {cleaned}\n")

            speak_text(response)

        except KeyboardInterrupt:
            print("\n👋 Voice dialogue interrupted.\n")
            m.clear_history.remote()
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
            continue


def main():
    m = model()
    print("🤖 Chat with the model (type 'quit' to exit, 'speech' for voice input, 'stt-tts' for voice dialogue)\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        elif prompt.lower() == "stt-tts":
            try:
                duration = input("Recording duration per turn in seconds (default 5): ").strip()
                duration = int(duration) if duration else 5
                voice_dialogue(m, duration=duration)
            except Exception as e:
                print(f"❌ Voice dialogue error: {e}\n")
            continue
        elif prompt.lower() == "speech":
            try:
                duration = input("Recording duration in seconds (default 5): ").strip()
                duration = int(duration) if duration else 5
                audio_bytes = record_audio(duration=duration)
                prompt = m.transcribe.remote(audio_bytes)
                if not prompt:
                    print("❌ Could not transcribe audio. Try again.\n")
                    continue
                print(f"🗣️  You said: {prompt}")
            except Exception as e:
                print(f"❌ Speech error: {e}\n")
                continue
        elif not prompt:
            continue

        response = m.chat.remote(prompt)
        print(f"\nAssistant: {clean_response(response)}\n")


if __name__ == "__main__":
    main()