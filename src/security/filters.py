import re

# https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html

class PromptInjectionFilter:
    def __init__(self):
        self.dangerous_patterns = [
            r'ignore\s+(all\s+)?previous\s+instructions?',
            r'you\s+are\s+now\s+(in\s+)?developer\s+mode',
            r'system\s+override',
            r'reveal\s+prompt',
        ]
        self.fuzzy_patterns = ['ignore', 'bypass', 'override', 'reveal', 'delete', 'system']

    def detect_injection(self, text: str) -> bool:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.dangerous_patterns):
            return True

        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            for pattern in self.fuzzy_patterns:
                if self._is_similar_word(word, pattern):
                    return True
        return False

    def _is_similar_word(self, word: str, target: str) -> bool:
        if len(word) != len(target) or len(word) < 3:
            return False
        return (word[0] == target[0] and
                word[-1] == target[-1] and
                sorted(word[1:-1]) == sorted(target[1:-1]))

    def sanitize_input(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(.)\1{3,}', r'\1', text)
        for pattern in self.dangerous_patterns:
            text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE)
        return text[:10000]