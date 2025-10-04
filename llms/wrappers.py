from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

class LLMWrapper:
    def __init__(self, backend="mock", llm=None, openai_model="gpt-3.5-turbo", openai_api_key=None):
        self.backend = backend
        if backend == "openai":
            self.llm = ChatOpenAI(
                model=openai_model,
                temperature=0,
                openai_api_key=openai_api_key
            )
        elif backend == "mock":
            self.llm = llm or (lambda prompt: '{"cleaned_data_summary": "mocked summary", "confidence": 0.9}')
        else:
            raise ValueError("Unknown backend")

    def __call__(self, prompt):
        if self.backend == "openai":
            # wrap prompt in a HumanMessage
            message = HumanMessage(content=prompt)
            response = self.llm([message])
            return response.content  # string output
        else:
            return self.llm(prompt)
