import modal

model = modal.Cls.from_name("example-inference", "Model")


def record_audio(duration=5, sample_rate=16000) -> bytes:
    """Record audio from the microphone and return raw WAV bytes."""
    import io
    import sounddevice as sd
    import soundfile as sf

    print(f"🎙️  Listening for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("✅ Recording complete. Sending to Modal for transcription...")

    # Convert to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def main():
    m = model()
    print("🤖 Chat with the model (type 'quit' to exit, 'speech' for voice input)\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
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
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()