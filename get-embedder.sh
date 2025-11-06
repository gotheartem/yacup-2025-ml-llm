mkdir -p models/Qwen3-Embedding-0.6B-GGUF
cd models/Qwen3-Embedding-0.6B-GGUF
rm -f Qwen3-Embedding-0.6B-GGUF
wget https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf
cd ../..