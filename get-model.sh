mkdir -p models/T-lite-it-1.0-Q8_0-GGUF
cd models/T-lite-it-1.0-Q8_0-GGUF
rm -f t-lite-it-1.0-q8_0.gguf
wget https://huggingface.co/t-tech/T-lite-it-1.0-Q8_0-GGUF/resolve/main/t-lite-it-1.0-q8_0.gguf
cd ../..