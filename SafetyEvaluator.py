from typing import List

class SafetyEvaluator:
    """
    This class is responsible for evaluating the safety of generated text using a keyword-based approach.
    """

    def __init__(self, bad_words: List[str]):
        self.bad_words = bad_words

    def evaluate_safety_keyword(self, generated_text: str) -> bool:
        """
        Evaluates the safety of the generated text using a simple keyword-based approach.
        Returns True if the text is safe, False otherwise.
        """
        for word in self.bad_words:
            if word in generated_text.lower():
                return False  # Unsafe content detected
        return True  # Text is safe

def main():
    # Initialize the safety evaluator with a list of bad words
    bad_words = ["hate", "violence", "profanity", "malicious"]
    evaluator = SafetyEvaluator(bad_words)

    # Example usage: Evaluate the safety of generated text
    generated_text = "Let's have a peaceful discussion."
    safety_result = evaluator.evaluate_safety_keyword(generated_text)
    if safety_result:
        print("The generated text is safe.")
    else:
        print("Unsafe content detected in the generated text.")

if __name__ == "__main__":
    main()
