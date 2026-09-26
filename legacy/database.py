from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from pinecone import PodSpec

load_dotenv(override=True)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "ipc186045"

def create_index():
    if not pc.has_index(index_name):
        print(f"Creating index {index_name}..")
        # pc.create_index_for_model(
        #     name=index_name,
        #     cloud="aws",
        #     region="us-east-1",
        #     embed={
        #         "model":"llama-text-embed-v2",
        #         "field_map":{"text": "chunk_text"}
        #     }
        # )
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index {index_name} is created :)")
    else:
        print(f"Index {index_name} already exists")