from backend.resume_parsing.parser import ResumeParser
import logging

logging.basicConfig(level=logging.INFO)

def test_parser():
    cwd = "x:\\Project" # Assuming running from project root
    print("Initializing parser...")
    try:
        parser = ResumeParser(model="qwen3:1.7b")
        print("Parser initialized.")
        
        text = """
        John Doe
        Software Engineer
        Email: john@example.com
        Experience:
        - Senior Developer at TechCorp (2020-Present)
        Skills: Python, Rust, AI
        """
        
        print("Parsing text...")
        data = parser.parse_text(text)
        print("Parsed Data:")
        print(data.model_dump_json(indent=2))
        print("SUCCESS")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_parser()
