FROM ghcr.io/astral-sh/uv:python3.12-alpine

# Set working directory
WORKDIR /app

# Copy project files
# COPY pyproject.toml uv.lock ./
# COPY main.py ./
# COPY README.md ./
COPY . .

# RUN uv pip install --system --all-extras
# Install dependencies
RUN uv sync


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
ENV TAVILY_API_KEY=${TAVILY_API_KEY:-""}


# Authenticate with UiPath during build
RUN uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL"

RUN uv run uipath run agent '{\"company_name\":\"uipath\"}'