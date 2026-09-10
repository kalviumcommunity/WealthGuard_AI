import json

# Mock OpenAI client for demonstration without requiring real API keys
class MockOpenAI:
    class chat:
        class completions:
            @staticmethod
            def create(model, messages, response_format=None, temperature=0):
                prompt = messages[-1]["content"]
                
                class Msg:
                    def __init__(self, c):
                        self.content = c
                class Choice:
                    def __init__(self, c):
                        self.message = Msg(c)
                class Response:
                    def __init__(self, c):
                        self.choices = [Choice(c)]
                
                # Simulate different model failure modes based on the prompt
                if "malformed" in prompt:
                    return Response('Here is your answer:\n{"answer": "It is 30 days", "source": "Policy"}')
                elif "missing" in prompt:
                    return Response('{"answer": "It is 30 days"}')
                else:
                    return Response('{"answer": "You have 30 days to request a refund.", "source": "Refund Policy v2"}')

client = MockOpenAI()

def ask_assistant(question, enforce_json=True, recovery_mode=False):
    """
    Task 1: Prompt for a defined JSON structure using JSON/response-format mode
    """
    SYSTEM = 'Reply with ONLY a JSON object: {"answer": string, "source": string}. No extra text.'
    
    if recovery_mode:
        SYSTEM += ' REMINDER: You MUST return valid JSON. Do not include markdown or prose.'

    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}
        ],
        "temperature": 0 # lower temperature for more deterministic formatting
    }
    
    if enforce_json:
        kwargs["response_format"] = {"type": "json_object"}
        
    return client.chat.completions.create(**kwargs).choices[0].message.content

def parse(raw, required=("answer", "source")):
    """
    Task 2 & 4: Parse into a usable object and validate required fields.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, "malformed JSON - could not decode"
        
    missing = [k for k in required if k not in data]
    if missing:
        return None, f"missing fields: {missing}"
        
    return data, None

def get_answer(question):
    print(f"--- Asking: '{question}' ---")
    raw = ask_assistant(question)
    print(f"Raw response: {raw}")
    
    # Task 3: Handle malformed JSON gracefully without crashing
    data, err = parse(raw)
    
    if err:
        print(f"[!] Error parsing response: {err}")
        print("[!] Attempting recovery (retrying with stricter prompt)...")
        
        # Strip the simulation flags to get a valid response on the retry
        retry_question = question.replace("malformed", "").replace("missing", "")
        raw_retry = ask_assistant(retry_question, recovery_mode=True)
        print(f"Retry raw response: {raw_retry}")
        
        data, err = parse(raw_retry)
        
        if err:
            print(f"[!] Recovery failed: {err}")
            return None
        else:
            print("[+] Recovery successful!")
            
    return data

if __name__ == "__main__":
    print("=== Structured Output & JSON Parsing Demo ===\n")
    
    # 1. Valid JSON case
    res1 = get_answer("What is the refund window?")
    print(f"Parsed Object: {res1}\n")
    
    # 2. Malformed JSON case
    res2 = get_answer("What is the refund window? (simulate malformed)")
    print(f"Parsed Object: {res2}\n")
    
    # 3. Missing fields case
    res3 = get_answer("What is the refund window? (simulate missing)")
    print(f"Parsed Object: {res3}\n")
