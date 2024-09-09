from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "GPT_doyoutrustme"

# 1. LLM 초기화
# ChatOpenAI 객체 생성
llm = ChatOpenAI(
    temperature=0.7,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o",  # 모델명
)
# 2. 주제 생성 체인
topic_prompt = ChatPromptTemplate.from_template(
    "주어진 Topic '{topic}'에 대해 5개의 세부 주제를 생성해주세요. 각 주제는 쉼표로 구분해 주세요."
)
topic_chain = topic_prompt | llm | CommaSeparatedListOutputParser()

# 3. 문제 생성 체인
question_prompt = ChatPromptTemplate.from_template(
    """다음 정보를 바탕으로 수능형 영어 문제를 생성해주세요:
    주제: {subtopic}
    난이도: {difficulty}
    문제 유형: {question_type}
    
    문제와 보기, 정답을 포함해 주세요."""
)
question_chain = question_prompt | llm

# 4. 전체 체인 구성
def generate_questions(inputs):
    subtopics = topic_chain.invoke({"topic": inputs["topic"]})
    questions = []
    for subtopic in subtopics:
        question = question_chain.invoke({
            "subtopic": subtopic,
            "difficulty": inputs["difficulty"],
            "question_type": inputs["question_type"]
        })
        questions.append(question)
    return questions

full_chain = RunnablePassthrough() | generate_questions

# 5. 체인 실행
result = full_chain.invoke({
    "topic": "Environmental Issues",
    "difficulty": "중",
    "question_type": "독해"
})

for i, question in enumerate(result, 1):
    print(f"Question {i}:")
    print(question)
    print("\n" + "="*50 + "\n")