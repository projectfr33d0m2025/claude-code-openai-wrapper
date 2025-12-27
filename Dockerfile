FROM python:3.12-slim

# Install system deps (curl for Poetry installer, Node.js for Claude CLI)
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x (required for Claude Code CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Verify installations
RUN node --version && npm --version && claude --version

# Install Poetry globally
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Copy the app code
COPY . /app

# Set working directory
WORKDIR /app

# Install Python dependencies with Poetry
RUN poetry install --no-root

# Expose the port (default 8000)
EXPOSE 8000

# Run the app with Uvicorn (production mode without reload for stability)
CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
