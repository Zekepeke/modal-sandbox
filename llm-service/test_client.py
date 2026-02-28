import modal

def test_llm():
    # Connect to the running Modal app
    f = modal.Function.lookup("h100-llama-service", "Model.generate")
    
    # Mock data like what your RAG function will send
    user_msg = "Where do I find the policy on X?"
    context = "Policy X is located in the Purdue Student Union, Room 204. [Image: A blue door with a gold sign]."

    print("LLM Response: ", end="")
    for chunk in f.remote_gen(user_prompt=user_msg, rag_context=context):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    test_llm()