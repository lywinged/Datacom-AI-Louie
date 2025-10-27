"""
Main entry point for the backend API
This file imports the FastAPI app from backend.main for Docker deployment
"""
from backend.main import app

__all__ = ["app"]
