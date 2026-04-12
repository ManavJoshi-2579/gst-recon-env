"""Development server entrypoint."""
import uvicorn

def run_dev_server():
    """Entry point for development server."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    run_dev_server()
