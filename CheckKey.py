# Imports
from langchain_openai import ChatOpenAI
import httpx

# Disable SSL verification (only for hackathon/demo – not recommended in production)
client = httpx.Client(verify=False)

# Initialize ChatOpenAI
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-wzADUHZAY9TDphuTQZfYQA",  # Replace with Hackathon-provided key
    http_client=client
)

# Test a query
response = llm.invoke("Hi")
print(response)