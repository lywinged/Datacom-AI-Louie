FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app:/app/backend/services
ENV ORT_NO_EXE_STACK_CHECK=1

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copy only the backend application package
COPY backend/backend /app/backend
COPY backend/backend/services/inference /app/inference_service
COPY backend/main.py /app/main.py

# Ensure the models directory exists (host volume will mount here)
RUN mkdir -p /app/models

# Clear executable stack requirement for onnxruntime shared library
RUN patchelf --clear-execstack \
    /usr/local/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so

EXPOSE 8001

# Health check endpoint for the inference service
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Launch the ONNX inference API
CMD ["python", "-m", "backend.services.inference.main_onnx"]
