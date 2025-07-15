FROM ghcr.io/astral-sh/uv:python3.12-alpine

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY main.py ./
COPY README.md ./

# Install dependencies
RUN uv pip install --system --all-extras

# Create environment file from build args
ARG CLIENT_ID
ARG CLIENT_SECRET
ARG BASE_URL

# Validate required environment variables
RUN if [ -z "$CLIENT_ID" ]; then echo "CLIENT_ID build arg is required" && exit 1; fi
RUN if [ -z "$CLIENT_SECRET" ]; then echo "CLIENT_SECRET build arg is required" && exit 1; fi
RUN if [ -z "$BASE_URL" ]; then echo "BASE_URL build arg is required" && exit 1; fi

# Set environment variables for runtime
ENV CLIENT_ID=$CLIENT_ID
ENV CLIENT_SECRET=$CLIENT_SECRET
ENV BASE_URL=$BASE_URL

# Authenticate with UiPath during build
RUN uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

# Create startup script
RUN cat > /app/startup.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting UiPath Retrieval Chain..."

# Re-authenticate in case token expired
echo "Authenticating with UiPath..."
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

# Check if arguments are provided
if [ $# -eq 0 ]; then
    echo "Running retrieval chain with default parameters..."
    exec uv run python main.py
else
    echo  "Running retrieval chain with parameters: $@"
    exec uv run python main.py "$@"
fi
EOF

RUN chmod +x /app/startup.sh

# Default command
CMD ["/app/startup.sh"]
