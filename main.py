
"""Top-level entrypoint for the tweet trading analysis app."""

from dotenv import load_dotenv

from src.app.ui import render_app


load_dotenv()


if __name__ == "__main__":
    render_app()
