import os
from google.adk.web import server

# Set the directory where your agents are located.
os.environ["AGENTS_DIR"] = "student"

if __name__ == "__main__":
    server.run()
