import os
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = """
You are a precise string reversal engine.
Given ANY input string, output ONLY the exact characters in strict reverse order.
- Treat the entire input as one continuous string, even if it looks like multiple words or protocols.
- Do NOT split by meaning, dictionary words, or known prefixes/suffixes (e.g., http, status, file extensions, domain parts).
- Preserve all characters and casing; do not insert or remove spaces or punctuation.
- Output must be a single sequence of characters with nothing else.

Examples (follow EXACTLY):
Q: Reverse the order of letters in the following word. Only output the reversed word, no other text:
cat
A: tac

Q: Reverse the order of letters in the following word. Only output the reversed word, no other text:
http
A: ptth

Q: Reverse the order of letters in the following word. Only output the reversed word, no other text:
status
A: sutats

Q: Reverse the order of letters in the following word. Only output the reversed word, no other text:
protocol
A: locotorp

Q: Reverse the order of letters in the following word. Only output the reversed word, no other text:
internet
A: tenretni

Q: Reverse the order of letters in the following word. Only output the reversed word, no other text:
apple
A: elppa

Q: Reverse the order of letters in the following word. Only output the reversed word, no other text:
httpstatus
A: sutatsptth
"""

USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""


EXPECTED_OUTPUT = "sutatsptth"

def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="mistral-nemo:12b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        output_text = response.message.content.strip()
        if output_text.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {output_text}")
    return False

if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)