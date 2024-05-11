from GenAiQuestionGenerator import *
from SafetyEvaluator import *

class TextSafetyIntegrityEvaluator:
    """
        This class is responsible for evaluating the integrity and safety of generated text, including questions and answers,
        using multiple approaches.
    """


    MIN_QUESTION_LENGTH = 5  # Minimum length for a valid question.
    MIN_ANSWER_LENGTH = 5  # Minimum length for a valid answer.
    INVALID_CHARS = {"$", "#", "%"}  # Invalid characters that should not appear in the text.


    def __init__(self, question_generator: GenAiQuestionGenerator, safety_evaluator: SafetyEvaluator):
        self.question_generator = question_generator
        self.safety_evaluator = safety_evaluator

    def evaluate_safety_and_integrity(self, generated_text: str) -> dict:
        """
            This function evaluates the safety and integrity of the generated text including questions and answers.
            Returns a dictionary with the safety evaluation and integrity check results.
        """
        safety_and_integrity_results = {}

        # Generate questions and answers based on the generated text:
        generated_answers = self.generate_answers(generated_text)

        # Evaluate safety and integrity of the generated answers:
        for question, answer in generated_answers.items():
            safety_and_integrity_results[question] = {
                "safety": self.safety_evaluator.evaluate_safety_with_blacklist(answer),
                "integrity": self.check_integrity(question, answer)
            }

        return safety_and_integrity_results


    def check_integrity(self, question: str, answer: str) -> bool:
        """
            This function checks the integrity of the generated question-answer pair.
            Returns True if both question and answer meet the integrity criteria, False otherwise.
        """
        # Check if both question and answer are non -empty:
        if not question.strip() or not answer.strip():
            return False

        # Check the length of the question and answer:
        if len(question) < self.MIN_QUESTION_LENGTH or len(answer) < self.MIN_ANSWER_LENGTH:
            return False

        # Check for the presence of specific characters or patterns that indicate integrity issues
        if any(char in self.INVALID_CHARS for char in question) or any(char in self.INVALID_CHARS for char in answer):
            return False

        # If all checks pass, consider the question- answer pair as intact:
        return True

    def generate_answers(self, generated_text: str) -> dict:
        """
            This function generates questions and answers based on the generated text.
            Returns a dictionary with questions as keys and generated answers as values.
        """
        generated_answers = {}

        # Example: Generate a question for the generated text:
        question = "What is your opinion about the following statement? Agree\ Disagree?"

        # Generate answers using the GenAiQuestionGenerator:
        generated_answer = "This is the generated answer for the question: " + generated_text

        generated_answers[question] = generated_answer

        return generated_answers

def main():
    # Initialize GenAiQuestionGenerator and SafetyEvaluator:
    api_key = "AIzaSyB-HcnrRqN9qopgoLC2HhedpQykv2h6HNE"
    train_file = "train.json"
    train_label_file = "train_label.json"
    is_google_search_available = False
    question_generator = GenAiQuestionGenerator(api_key, train_file, train_label_file, is_google_search_available)
    safety_evaluator = SafetyEvaluator()

    # Initialize TextSafetyIntegrityEvaluator:
    text_safety_integrity_evaluator = TextSafetyIntegrityEvaluator(question_generator, safety_evaluator)

    # Example usage: Evaluate the safety and integrity of a model's generated text:
    generated_text = "Let's have a peaceful discussion."
    safety_and_integrity_results = text_safety_integrity_evaluator.evaluate_safety_and_integrity(generated_text)

    # Display safety evaluation and integrity check results:
    print("Safety evaluation and integrity check results:")
    for question, results in safety_and_integrity_results.items():
        safety_result = results["safety"]
        integrity_result = results["integrity"]
        print(f"Question: {question}")
        print("Safety evaluation:", "Safe" if safety_result else "Unsafe")
        print("Integrity check:", "Passed" if integrity_result else "Failed")
        print()

if __name__ == "__main__":
    main()
