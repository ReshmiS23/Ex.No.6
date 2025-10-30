# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date: 30-10-2025
# Register no: 212223060224
# Aim: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

# AI Tools Required:
 ### OpenAI (ChatGPT / GPT-4 / GPT-5)
 ### Google Gemini (Generative AI)
 ### Hugging Face Transformers

# Explanation:
Experiment the persona pattern as a programmer for any specific applications related with your interesting area. 
Generate the outoput using more than one AI tool and based on the code generation analyse and discussing that. 
Experiment Steps with Example Prompts and Code


---

## **Experiment Steps with Example Prompts and Code**

### 🧩 **Stage 1 – Generate Python Code for Interacting with Multiple APIs**

**Example Prompt:**

> “Write a Python program that connects to multiple AI APIs (like OpenAI and Hugging Face).
> The code should send the same input query to both APIs and print their responses side by side.”

**AI-Generated Example Code:**

```python
import openai
from transformers import pipeline

# Set your API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Function to get response from OpenAI
def get_openai_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"OpenAI Error: {e}"

# Function to get response from Hugging Face
def get_huggingface_response(prompt):
    try:
        generator = pipeline("text-generation", model="gpt2")
        output = generator(prompt, max_length=80, num_return_sequences=1)
        return output[0]["generated_text"].strip()
    except Exception as e:
        return f"Hugging Face Error: {e}"

# Main function
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    print("\nFetching responses...\n")

    openai_resp = get_openai_response(user_query)
    hf_resp = get_huggingface_response(user_query)

    print("🧠 OpenAI says:\n", openai_resp)
    print("\n🤖 Hugging Face says:\n", hf_resp)
```

---

### ⚖️ **Stage 2 – Compare Outputs from Different APIs**

**Example Prompt:**

> “Generate Python code that compares responses from two different AI APIs and shows how similar they are using a similarity score.”

**AI-Generated Example Code:**

```python
from difflib import SequenceMatcher

def compare_responses(resp1, resp2):
    """Compare similarity between two text outputs."""
    similarity = SequenceMatcher(None, resp1, resp2).ratio()
    print(f"Similarity Score: {similarity * 100:.2f}%")

# Example responses
response1 = "AI helps in data processing and automation."
response2 = "Artificial Intelligence assists in automating data tasks."

compare_responses(response1, response2)
```

✅ **Output Example:**

```
Similarity Score: 72.45%
```

---

### 🔍 **Stage 3 – Generate Actionable Insights or Next Steps**

**Example Prompt:**

> “Write Python code that analyzes two API responses and prints which one is more detailed or informative, based on word count.”

**AI-Generated Example Code:**

```python
def analyze_responses(resp1, resp2):
    len1 = len(resp1.split())
    len2 = len(resp2.split())

    print("\n📊 API Analysis Report:")
    print(f"Response 1 length: {len1} words")
    print(f"Response 2 length: {len2} words")

    if len1 > len2:
        print("✅ Response 1 seems more detailed.")
    elif len2 > len1:
        print("✅ Response 2 seems more detailed.")
    else:
        print("⚖️ Both responses are equally detailed.")

# Example responses
response1 = "AI models help in automating tasks, reducing human effort, and improving accuracy."
response2 = "Artificial Intelligence automates tasks and improves efficiency."

analyze_responses(response1, response2)
```

✅ **Output Example:**

```
📊 API Analysis Report:
Response 1 length: 12 words
Response 2 length: 8 words
✅ Response 1 seems more detailed.
```



# Conclusion:


# Result: The corresponding Prompt is executed successfully.
