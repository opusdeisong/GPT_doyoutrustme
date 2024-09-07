from dotenv import load_dotenv
load_dotenv()

# 프로젝트 이름을 변경하고 싶다면:
import os
os.environ["LANGCHAIN_PROJECT"] = "GPT_doyoutrustme"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ChatOpenAI 모델 초기화
model = ChatOpenAI()

# 프롬프트 템플릿 생성
prompt = PromptTemplate.from_template("{topic}에 대해 3문장으로 설명해줘.")

# 체인 구성
chain = prompt | model | StrOutputParser()

# 체인 실행
result = chain.invoke({"topic": "인공지능"})
print(result)
