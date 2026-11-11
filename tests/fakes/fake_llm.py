from src.ports.llm_port import LLMPort, LLMResponse, ModelPricing, TokenUsage


class FakeLLM(LLMPort):
    """Scripted LLMPort: returns `answers` in order (an Exception instance is raised instead)."""

    def __init__(self, answers=None, pricing=ModelPricing(0.562, 2.25, 0.141), usage=TokenUsage(100, 20)):
        self.answers = list(answers or [])
        self._pricing = pricing
        self.usage = usage
        self.requests = []

    @property
    def name(self):
        return "fake-model"

    @property
    def pricing(self):
        return self._pricing

    def generate_json(self, request):
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else {}
        if isinstance(answer, Exception):
            raise answer
        return LLMResponse(data=answer, usage=self.usage, model="fake-model")
