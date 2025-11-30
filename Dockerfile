# =============================================================================
# Speed Climbing Performance Analysis - Dockerfile
# Multi-stage build optimized for Coolify deployment
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install Python dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements_core.txt .
RUN pip install --no-cache-dir --user -r requirements_core.txt

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal runtime image
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS production

# Install runtime dependencies
# - wget: Required for health checks (Coolify requirement)
# - libgl1: OpenCV dependency (replaces deprecated libgl1-mesa-glx)
# - libglib2.0-0: OpenCV dependency
# - libsm6, libxext6, libxrender1: Additional OpenCV GUI dependencies
# - libegl1, libgles2: EGL/OpenGL ES for MediaPipe CPU fallback
# - ffmpeg: Video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libegl1 \
    libgles2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN addgroup --gid 1001 appgroup && \
    adduser --uid 1001 --gid 1001 --disabled-password --gecos '' appuser

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY --chown=appuser:appgroup speed_climbing/ ./speed_climbing/
COPY --chown=appuser:appgroup scripts/ ./scripts/
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup examples/ ./examples/
COPY --chown=appuser:appgroup quick_start.py ./
COPY --chown=appuser:appgroup requirements_core.txt ./

# Create data directory
RUN mkdir -p /app/data && chown -R appuser:appgroup /app/data

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check for Coolify
# Uses wget (installed above) to check Streamlit health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8501/_stcore/health || exit 1

# Environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# MediaPipe/OpenCV configuration for headless Docker
# These allow GPU if available, fallback to CPU otherwise
# EGL platform for headless rendering (works with or without GPU)
ENV PYOPENGL_PLATFORM=egl
# If you have GPU issues, uncomment these to force CPU-only:
# ENV CUDA_VISIBLE_DEVICES=""
# ENV MEDIAPIPE_DISABLE_GPU=1
# ENV LIBGL_ALWAYS_SOFTWARE=1

# Start Streamlit application
# Default: New user-facing analysis app (Phase 5)
# For review interface: Override with scripts/review_interface/app.py
CMD ["python", "-m", "streamlit", "run", "scripts/analysis_app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
