from langchain_ollama import OllamaEmbeddings, ChatOllama

class Models:
    def __init__(self, model="llama", embed="large"):
        ## Retrieval
        if embed == "large":
            # Embedding ollama (mxbai-embed-large)
            self.embeddings_ollama = OllamaEmbeddings(
                model="mxbai-embed-large"
            )

        else:
            # Embedding ollama (all-minilm)
            self.embeddings_ollama = OllamaEmbeddings(
                model="all-minilm"
            )


        ## Generation
        if model == "llama":
            # Model ollama (llama3.2)
            self.model_ollama = ChatOllama(
                model="llama3.2",
                temperature=0,
            )
        
        else:
            # Model ollama (mistral)
            self.model_ollama = ChatOllama(
                model="mistral",
                temperature=0,
            )
