FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

COPY . .

# Install dependencies for all testcases
RUN uv sync


# Set environment variables for runtime
ENV TAVILY_API_KEY=${TAVILY_API_KEY:-""}
ENV UIPATH_TENANT_ID=${UIPATH_TENANT_ID:-""}
ENV UIPATH_JOB_KEY=3a03d5cb-fa21-4021-894d-a8e2eda0afe0

CMD ["bash"]
