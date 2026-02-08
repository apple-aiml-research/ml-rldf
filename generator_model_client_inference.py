#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import requests
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--description", type=str, required=True, help="Description of the UI to generate"
)
parser.add_argument(
    "--model",
    type=str,
    default="rldfcoder-qwen3-sketch",
    help="Model name registered on server",
)
parser.add_argument(
    "--url",
    type=str,
    default="http://localhost:11434/api/generate",
    help="Hosted API endpoint",
)
args = parser.parse_args()

# Prompt template for UI generation
prompt_prefix = "provide the complete HTML code for a web page implemented with only tailwind CSS and font awesome icons. do not use any templating languages like jinja. the result should resemble an award-winning iOS app. include realistic and complete placeholder data. do not treat this as the starting point for an app - it should be the mockup of a final complete UI. remember to include alt text for all images. do not use javascript. do not use SVGs. here is a description of the webpage: "

# Construct full prompt
full_prompt = prompt_prefix + args.description

# Prepare request data
data = {
    "model": args.model,
    "stream": False,
    "prompt": full_prompt,
    "options": {"num_predict": 4096},
}

# Send request to server
print(f"Generating UI for: {args.description}")
print("Waiting for response...\n")

try:
    response = requests.post(args.url, json=data, timeout=30)
    response.raise_for_status()
    result = response.json()["response"]
    print("Generated HTML:")
    print("=" * 80)
    print(result)
    print("=" * 80)
except requests.RequestException as e:
    print(f"Network request failed: {e}")
    exit(1)
except KeyError:
    print("Invalid response format: missing 'response' key")
    exit(1)
except ValueError as e:
    print(f"JSON parsing failed: {e}")
    exit(1)
