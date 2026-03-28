import re
from typing import Literal
from groq import Groq
import config
from utils.logger import get_logger

logger = get_logger(__name__)

Route = Literal["math", "rag", "memory", "general", "search"]

_MATH_KEYWORDS = {
    "calculate", "compute", "solve", "equation", "formula",
    "add", "subtract", "multiply", "divide", "sum", "average",
    "percentage", "derivative", "integral", "square root",
    "what is", "+", "-", "*", "/", "%", "^",
}
_MATH_REGEX = re.compile(r"\b\d+\s*[\+\-\*\/\^]\s*\d+")

_MEMORY_KEYWORDS = {
    "earlier", "before", "previously", "last time", "remember",
    "what did we", "what did i", "you said", "we discussed",
    "recall", "history", "our conversation", "mentioned",
}

_SEARCH_KEYWORDS = {
    "today", "now", "current", "latest", "recent", "news", "president",
    "weather", "live", "stock price", "football score", "real time"
}

class RouterAgent:
    ROUTER_PROMPT = """You are a query router for a multi-agent chatbot.
Your job is to classify the user's query into EXACTLY ONE of these categories:

  math    → The query requires mathematical calculation or numerical reasoning.
  rag     → The query asks about content from uploaded PDF documents specifically.
  memory  → The query refers to previous conversation history or what was discussed before.
  search  → The query asks for real-time or very current information like news, 
            current world leaders (who is the president?), today's weather/date,
            live sports scores, or recently published facts.
  general → The query asks for general knowledge, definitions, history (pre-2023),
            science, and anything NOT in uploaded docs or needing a live search.

Rules:
- Reply with ONLY the single word: math, rag, memory, search, or general
- No explanation, no punctuation, no other text
- If unsure between 'search' and 'general', choose 'search' for current news or people
- Only use 'rag' if the user clearly refers to an uploaded document

User query: "{query}"
Category:"""

    def __init__(self):
        self._client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("RouterAgent initialised.")

    def _llm_classify(self, query: str) -> Route | None:
        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=[
                    {"role": "user", "content": self.ROUTER_PROMPT.format(query=query)}
                ],
                temperature=0.0,
                max_tokens=5,
            )
            label = response.choices[0].message.content.strip().lower()

            if label in ("math", "rag", "memory", "general", "search"):
                logger.info(f"Router (LLM): '{query[:50]}' → {label}")
                return label
            else:
                logger.warning(f"Router (LLM) returned unexpected label: '{label}'")
                return None

        except Exception as exc:
            logger.error(f"Router LLM call failed: {exc}. Falling back to rules.")
            return None

    def _rule_classify(self, query: str) -> Route:
        q_lower = query.lower()

        if _MATH_REGEX.search(query):
            logger.info(f"Router (rules/regex): → math")
            return config.ROUTE_MATH

        if any(kw in q_lower for kw in _MATH_KEYWORDS):
            logger.info(f"Router (rules/keywords): → math")
            return config.ROUTE_MATH

        if any(kw in q_lower for kw in _MEMORY_KEYWORDS):
            logger.info(f"Router (rules/keywords): → memory")
            return "memory"

        if any(kw in q_lower for kw in _SEARCH_KEYWORDS):
            logger.info(f"Router (rules/keywords): → search")
            return "search"

        logger.info(f"Router (rules/default): → general")
        return "general"

    def route(self, query: str) -> Route:
        llm_route = self._llm_classify(query)

        if llm_route is not None:
            return llm_route

        return self._rule_classify(query)
