import json
import random
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from googlesearch import search
from bs4 import BeautifulSoup

class GenAiQuestionGenerator(object):
    """
        This class is designed to facilitate the generation of questions and answers using generative AI models.
        It allows users to input questions and options, then prompts the AI model to generate answers based on the provided inputs.
    """


    def __init__(self, api_key, train_file, train_label_file, is_google_search_available):
        self._api_key = api_key
        self._train_file = train_file
        self._train_label_file = train_label_file
        self._is_google_search_available = is_google_search_available
        self._gen_ai_model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self._api_key,
        )


    @property
    def is_google_search_available(self):
        return self._is_google_search_available


    @is_google_search_available.setter
    def is_google_search_available(self, is_search_available):
        self._is_google_search_available = is_search_available


    @staticmethod
    def _load_from_json(file_path):
        """
            This function gets a path to a JSON file and returns the content of the JSON file as a list.
        """
        with open(file_path, 'r') as file:
            # For each line parse a valid JSON string and convert it into a python list:
            return [json.loads(line) for line in file]

    @staticmethod
    def _check_model_answer(model_answer, answer):
        """
            This function checks if the model answer matches the correct answer.
        """
        return (("<answer>a</answer>" in model_answer and not answer)
                or ("<answer>b</answer>" in model_answer and answer))


    @staticmethod
    def _search_google(search_term):
        """
            This function searches Google for the given term and generates an answer based on the search result.
        """
        search_results = list(search(search_term, num=1, stop=1, pause=2))
        if search_results:
            first_result_url = search_results[0]
            search_result_html = requests.get(first_result_url).text
            soup = BeautifulSoup(search_result_html, 'html.parser')
            return "Search result: " + soup.get_text(separator='\n')
        else:
                return "No relevant information found on the web."


    def _sample_questions_and_answers(self):
        questions_list = self._load_from_json(self._train_file)
        answers_list = self._load_from_json(self._train_label_file)
        questions_and_answers_list = list(enumerate(zip(answers_list, questions_list)))
        # Reorganize the order of the items:
        return random.sample(questions_and_answers_list, k=50)


    def _generate_answer(self, question_and_options):
        """
            This function prompts the generative AI model to generate an answer based on the provided question and options.
        """
        if self._is_google_search_available:
            message = ("I will give you a question or sentence to complete and two possible answers. "
                       "Please answer either A or B, depending on which answer is better. "
                       "You may write down your reasoning but please write your final answer (either A or B) "
                       "between the <answer> and </answer> tags. "
                       "The question is: " + question_and_options['goal'] +
                       "The first options is: " + question_and_options['sol1'] +
                       "The second options is: " + question_and_options['sol2'] +
                       "\n You can also return a search term between <search> and </search> tags instead of an answer, "
                       "and then receive the content of the first result in that Google search.")
        else:
            message = ("I will give you a question or sentence to complete and two possible answers. "
                       "Please answer either A or B, depending on which answer is better. "
                       "You may write down your reasoning but please write your final answer (either A or B) "
                       "between the <answer> and </answer> tags. "
                       "The question is: " + question_and_options['goal'] +
                       "The first options is: " + question_and_options['sol1'] +
                       "The second options is: " + question_and_options['sol2'])
        model_response = self._gen_ai_model.invoke(message)
        model_answer = model_response.content
        if self._is_google_search_available and "<search>" in model_answer:
            search_term = model_answer.split("<search>")[1].split("</search>")[0]
            search_result = self._search_google(search_term)
            return self._generate_answer_after_search(question_and_options, search_result)
        return model_answer


    def _review_answer(self, question, model_answer):
        """
            This function prompts a reviewer to assess the accuracy of the answer generated by the first model.
            The reviewer is presented with the question, options, and the answer generated by the model, along with its reasoning.
            The reviewer is instructed to indicate agreement by inserting the word 'correct' between <review> and </review> tags.
        """

        # Prompt the reviewer for review:
        message = ("I will give you a question or sentence to complete and two possible answers."
            "I will give you an answer, either A or B between the <answer> and </answer> tags, "
            "and its reasoning. Please review the question and the answer that was given "
            "and insert between <review> and </review> tags the word correct only when you agree."
            "The question is: " + question['goal'] +
            "The first options is: " + question['sol1'] +
            "The second options is: " + question['sol2'] +
            "The answer is: " + model_answer)
        response = self._gen_ai_model.invoke(message)
        return response.content == "<review>correct</review>"


    def _generate_answer_after_search(self, question_and_options, search_result):
        """
            This function prompts the generative AI model to generate an answer based on the provided question and options,
            taking into account additional information from a Google search result. The question, options, and search result
            are incorporated into the message passed to the model for answer generation.
        """

        message = ("I will give you a question or sentence to complete and two possible answers."
                   "Please answer either A or B, depending on which answer is better."
                   "You may write down your reasoning but please write your final answer (either A or B) "
                   "between the <answer> and </answer> tags."
                   "The question is: " + question_and_options['goal'] +
                   "The first options is: " + question_and_options['sol1'] +
                   "The second options is: " + question_and_options['sol2'] +
                   "\n You are also given the content of the first result in Google search, "
                   "of a term generated by another model to increase the chance of success: " + search_result)
        model_response = self._gen_ai_model.invoke(message)
        return model_response.content


    @property
    def success_rate(self):
        """
            This function creates a dictionary of 50 random answer & question pairs and calculates the model's success rate.
            It returns the percentage of the success rate.
        """
        success_rate = 0
        questions_and_answers = self._sample_questions_and_answers()
        # Calculate the success rate:
        for _, (answer, question) in questions_and_answers:
            model_answer = self._generate_answer(question).strip().lower() # Lowercased without whitespaces.
            if self._check_model_answer(model_answer, answer):
                success_rate += 1
        # Return the percentage of the success rate:
        return success_rate * 2


    @property
    def success_rate_with_review(self):
        """
            This function returns the percentage of the success rate of 50 random questions,
            answered identically by 2 independently models.
        """
        success_rate = 0
        questions_and_answers = self._sample_questions_and_answers()
        for _, (answer, question) in questions_and_answers:
            model_answer = self._generate_answer(question).strip().lower()
            # Review the answer (True\ False):
            review_result = self._review_answer(question, model_answer)
            # Keep reviewing until the reviewer agrees with the model's answer:
            while not review_result:
                model_answer = self._generate_answer(question).strip().lower()
                review_result = self._review_answer(question, model_answer)
            if self._check_model_answer(model_answer, answer):
                success_rate += 1
        return success_rate * 2

    def generate_answers(self, generated_text: str) -> dict:
        """
            This function generates questions and answers based on the generated text.
            Returns a dictionary with questions as keys and generated answers as values.
        """
        generated_answers = {}

        # Example: Generate a question for the generated text:
        question = "What is your opinion about the following statement? Agree\ Disagree?"

        # Generate answers using the GenAiQuestionGenerator:
        generated_answers[question] = self._generate_answer(question)

        return generated_answers

def main():
    api_key = "AIzaSyB-HcnrRqN9qopgoLC2HhedpQykv2h6HNE"
    train_file = "train.json"
    train_label_file = "train_label.json"
    gen_ai_question_generator = GenAiQuestionGenerator(api_key, train_file, train_label_file, False)

    # Part 1:
    success_rate = gen_ai_question_generator.success_rate
    print("********** Part 1 **********")
    print("We achieved " + str(success_rate) + "% success rate.")
    # Part 2:
    print("********** Part 2 **********")
    success_rate_with_review = gen_ai_question_generator.success_rate_with_review
    print("We achieved " + str(success_rate_with_review) + "% success rate, using another model as a reviewer.")
    print("Success rate has improved! Great Success!") if success_rate_with_review > success_rate else print ("Success rate has not improved.")
    # Part 3:
    print("********** Part 3 **********")
    gen_ai_question_generator.is_google_search_available = True
    success_rate_with_search = gen_ai_question_generator.success_rate
    print("We achieved " + str(success_rate_with_search) + "% success rate with google search available.")
    print("Success rate has improved! Great Success!") if success_rate_with_search > success_rate else print ("Success rate has not improved.")


if __name__ == "__main__":
    main()
