import sys
import os

# Create absolute path so src can be accessed correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.web.app import create_demo

# Need to assign to a variable named exactly 'demo' or 'app' for Hugging Face Spaces to detect
demo = create_demo()

if __name__ == "__main__":
    demo.launch()
