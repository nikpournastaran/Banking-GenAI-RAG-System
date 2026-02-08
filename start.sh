#!/bin/bash
set -e

echo "Starting RAG Chat Bot application..."
echo "Environment: $([ -n "$RENDER" ] && echo "Render" || echo "Local")"
echo "Python version: $(python --version)"

# Ensure data directory exists
if [ ! -d "/data" ]; then
    echo "Creating data directory..."
    mkdir -p /data
    chmod 777 /data
fi

# Check if index exists in persistent storage
if [ ! -f "/data/index.faiss" ] && [ -f "./index/index.faiss" ]; then
    echo "Index not found in persistent storage. Copying from local directory..."
    cp -rv ./index/* /data/
    echo "Index copied successfully to persistent storage."
fi

# Print diagnostics
echo "Checking directories:"
echo "- Current directory: $(ls -la)"
echo "- Data directory: $(ls -la /data)"
echo "- Available disk space: $(df -h /data)"

# Check API keys
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY environment variable is not set!"
else
    echo "OPENAI_API_KEY is set (not shown for security)"
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "WARNING: ANTHROPIC_API_KEY environment variable is not set!"
else
    echo "ANTHROPIC_API_KEY is set (not shown for security)"
fi

# Check Telegram bot token
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "WARNING: TELEGRAM_BOT_TOKEN environment variable is not set! Telegram bot will not run."
else
    echo "TELEGRAM_BOT_TOKEN is set (not shown for security). Telegram bot will run."
fi

# Start the application with proper error handling
echo "Starting web server and Telegram bot..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --proxy-headers