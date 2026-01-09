from vectore_store import VectorStoreManager
import os

print("=== BEFORE ===")
vs = VectorStoreManager()
print(f"Docs: {vs.collection.count()}")
print(f"Folder: {os.listdir('./chroma_db')}")

print("\n=== WIPING ===")
result = vs.clear_document()
print(result)

print("\n=== AFTER ===")
print(f"Docs: {vs.collection.count()}")
print(f"Folder: {os.listdir('./chroma_db')}")
