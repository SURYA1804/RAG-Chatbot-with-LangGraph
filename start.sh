#!/bin/bash
echo "Starting RAG Chatbot..."

# Start backend (FastAPI)
python backend/app.py &
echo "Backend started (PID: $!)"

# Start frontend (Streamlit) on Render's PORT
streamlit run frontend/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 &
echo "Frontend started (PID: $!)"

# Wait for both
wait
