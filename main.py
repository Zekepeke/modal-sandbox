import modal

model = modal.Cls.from_name("example-inference", "Model")


def main():
    m = model()
    print("🤖 Chat with the model (type 'quit' to exit)\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if not prompt:
            continue

        response = m.chat.remote(prompt)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()