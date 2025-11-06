FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install -U llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122/
COPY ./docker-requirements.txt ./docker-requirements.txt
RUN pip3 install -r docker-requirements.txt

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /workspace
COPY . .

# Use this for submission
# ENTRYPOINT ["python", "solution.py"]

# Use this for local testing
ENTRYPOINT ["bash", "test-running.sh"]