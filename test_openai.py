import os
import json
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key found: {bool(api_key)}")

try:
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Test API connection
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Return a simple JSON with a welcome message."}
        ],
        response_format={"type": "json_object"}
    )
    
    # Print response
    result = response.choices[0].message.content
    print("API Response:")
    print(result)
    print("\nTest successful! OpenAI integration is working.")
    
except Exception as e:
    print(f"Error: {str(e)}")
