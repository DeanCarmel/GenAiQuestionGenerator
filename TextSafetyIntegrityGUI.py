import tkinter as tk
from tkinter import scrolledtext
from TextSafetyIntegrityEvaluator import TextSafetyIntegrityEvaluator
from GenAiQuestionGenerator import GenAiQuestionGenerator
from SafetyEvaluator import SafetyEvaluator


class TextSafetyIntegrityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Safety Integrity Evaluator")

        # Initialize TextSafetyIntegrityEvaluator components
        api_key = "AIzaSyB-HcnrRqN9qopgoLC2HhedpQykv2h6HNE"
        train_file = "train.json"
        train_label_file = "train_label.json"
        is_google_search_available = False
        question_generator = GenAiQuestionGenerator(api_key, train_file, train_label_file, is_google_search_available)
        safety_evaluator = SafetyEvaluator()
        self.evaluator = TextSafetyIntegrityEvaluator(question_generator, safety_evaluator)

        # Create GUI elements
        self.label = tk.Label(root, text="Enter generated text:")
        self.label.pack()
        self.textbox = scrolledtext.ScrolledText(root, width=50, height=10)
        self.textbox.pack()
        self.button = tk.Button(root, text="Evaluate", command=self.evaluate)
        self.button.pack()
        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

    def evaluate(self):
        generated_text = self.textbox.get("1.0", tk.END)
        results = self.evaluator.evaluate_safety_and_integrity(generated_text)
        self.display_results(results)

    def display_results(self, results):
        self.result_label.config(text="Safety evaluation and integrity check results:")
        result_text = ""
        for question, data in results.items():
            safety_result = "Safe" if data["safety"] else "Unsafe"
            integrity_result = "Passed" if data["integrity"] else "Failed"
            result_text += f"Question: {question}\nSafety evaluation: {safety_result}\nIntegrity check: {integrity_result}\n\n"
        self.result_label.config(text=result_text)


def main():
    root = tk.Tk()
    app = TextSafetyIntegrityGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
