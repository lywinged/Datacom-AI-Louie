"""
Main entry point for the backend API
Re-export the FastAPI app for Docker/uvicorn to find
"""
import sys
import os

# Add parent directory to path so we can import backend.backend
sys.path.insert(0, os.path.dirname(__file__))

from backend.main import app

__all__ = ["app"]
