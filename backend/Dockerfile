FROM public.ecr.aws/lambda/python:3.11

# --- 1. Environment Configuration ---
# Tell FastEmbed/FlashRank where to find the 'baked' models
ENV FASTEMBED_CACHE_PATH=/var/task/fastembed_cache
ENV TEXT_EMBEDDING_MODEL_CACHE_DIR=/var/task/fastembed_cache

# --- 2. Install Python Dependencies ---
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- 3. Bake Models (Crucial Step) ---
COPY backend/scripts/download_models.py .
RUN python download_models.py

# --- 4. Copy Application Code ---
# Structure inside container: /var/task/backend/...
COPY backend/ ./backend/
COPY config/ ./config/
COPY utils/ ./utils/

# --- 5. Handover to Lambda ---
CMD ["backend.main.handler"]