import json
from nltk.stem import WordNetLemmatizer


class QuestionsData(object):
    def __init__(self, qPath, ansPath):
        self.__ques_path, self.__ann_path = qPath, ansPath

    def read_questions_data(self):
        with open(self.__ques_path, 'r') as f:
            json_obj = json.load(f)

        questions_info, questions = {}, []
        for question in json_obj["questions"]:
            img_id = question["image_id"]
            ques_id = question["question_id"]
            questions_info[question["question"]] = (img_id, ques_id)
            wordnetlemmtizer=WordNetLemmatizer()
            question["question"]=wordnetlemmtizer.lemmatize(question["question"].lower(),pos='v')
            questions.append(question["question"])
            print(WordNetLemmatizer.lemmatize(question["question"].lower(),pos='v'))
        return questions_info, questions

    def get_answers(self):
        with open(self.__ann_path, 'r') as f:
            json_obj = json.load(f)

        answers = {}
        for ann in json_obj["annotations"]:
            ques_id = ann["question_id"]
            list_of_answers = [item["answer"] for item in ann["answers"]]
            answers[ques_id] = list_of_answers
        return answers

#main

# questions_reader = QuestionsData("data/vqa/Questions/v2_OpenEnded_mscoco_test-dev2015_questions.json", "data/vqa/Annotations/v2_mscoco_val2014_annotations.json")
# #questions_info, questions = questions_reader.read_questions_data()
#
# print(questions_reader.get_answers())
