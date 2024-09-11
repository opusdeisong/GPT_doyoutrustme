import os
from dotenv import load_dotenv
from langchain.utilities import GoogleSerperAPIWrapper
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# Rich 라이브러리를 사용하여 콘솔 출력을 더 아름답게 만듭니다.
console = Console()

# .env 파일에서 환경 변수를 로드합니다. 이는 API 키를 안전하게 관리하기 위함입니다.
load_dotenv()

# 환경 변수에서 API 키를 가져옵니다.
serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# API 키가 제대로 설정되었는지 확인합니다.
if not serper_api_key or not openai_api_key:
    raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# Google Serper API를 사용하여 웹 검색을 수행할 수 있는 도구를 초기화합니다.
search = GoogleSerperAPIWrapper()
# OpenAI의 ChatGPT 모델을 초기화합니다. temperature=0으로 설정하여 일관된 출력을 얻습니다.
llm = ChatOpenAI(temperature=0)

# 초기 검색 및 분석을 위한 프롬프트 템플릿을 정의합니다.
initial_search_template = """
다음 질문에 대한 검색 결과를 분석하고 요약해주세요:
질문: {query}

검색 결과:
{search_result}

요약:
"""
initial_search_prompt = ChatPromptTemplate.from_template(initial_search_template)

# 초기 검색 및 분석을 위한 체인을 구성합니다.
# 이 체인은 쿼리를 받아 검색을 수행하고, 그 결과를 분석하여 요약합니다.
initial_chain = (
    {"query": RunnablePassthrough(), "search_result": lambda x: search.run(x)}
    | initial_search_prompt
    | llm
    | StrOutputParser()
)

# 추가 검색 필요성을 판단하기 위한 프롬프트 템플릿을 정의합니다.
need_more_search_template = """
다음 정보를 바탕으로, 추가 검색이 필요한지 판단해주세요:

질문: {query}
초기 분석: {initial_analysis}

추가 검색이 필요하면 "Yes", 그렇지 않으면 "No"라고 답해주세요.
결정:
"""
need_more_search_prompt = ChatPromptTemplate.from_template(need_more_search_template)

# 추가 검색 필요성을 판단하는 체인을 구성합니다.
need_more_search_chain = need_more_search_prompt | llm | StrOutputParser()

# 최종 답변을 생성하기 위한 프롬프트 템플릿을 정의합니다.
final_answer_template = """
다음 정보를 바탕으로 최종 응답을 생성해주세요:

질문: {query}
초기 분석: {initial_analysis}
추가 검색 결과: {additional_search}

최종 응답:
"""
final_answer_prompt = ChatPromptTemplate.from_template(final_answer_template)

# 최종 답변을 생성하는 체인을 구성합니다.
final_answer_chain = final_answer_prompt | llm | StrOutputParser()

def perplexity_style_search(query):
    # 사용자의 질문을 콘솔에 출력합니다.
    console.print(Panel(f"[bold cyan]질문:[/bold cyan] {query}", expand=False))

    # 초기 검색 및 분석을 수행합니다.
    with console.status("[bold green]초기 검색 및 분석 중...[/bold green]"):
        initial_analysis = initial_chain.invoke(query)
    console.print(Panel(Text(f"초기 분석:\n{initial_analysis}", style="green"), expand=False))

    # 추가 검색이 필요한지 판단합니다.
    with console.status("[bold yellow]추가 검색 필요성 판단 중...[/bold yellow]"):
        need_more = need_more_search_chain.invoke({"query": query, "initial_analysis": initial_analysis})
    console.print(f"[bold magenta]추가 검색 필요:[/bold magenta] {need_more}")

    # 필요한 경우 추가 검색을 수행합니다.
    if need_more.strip().lower() == "yes":
        with console.status("[bold blue]추가 검색 수행 중...[/bold blue]"):
            additional_search = search.run(f"More details about {query}")
        console.print("[bold blue]추가 검색 완료[/bold blue]")
    else:
        additional_search = "추가 검색 불필요"

    # 최종 응답을 생성합니다.
    with console.status("[bold red]최종 응답 생성 중...[/bold red]"):
        final_answer = final_answer_chain.invoke({
            "query": query,
            "initial_analysis": initial_analysis,
            "additional_search": additional_search
        })

    # 최종 응답을 콘솔에 출력합니다.
    console.print(Panel(Text(f"최종 응답:\n{final_answer}", style="bold white on blue"), expand=False))

    return final_answer

# 메인 루프 함수입니다.
def main():
    # 프로그램 시작 메시지를 출력합니다.
    console.print(Panel("[bold green]AI 검색 시스템에 오신 것을 환영합니다![/bold green]\n'종료'를 입력하면 프로그램이 종료됩니다.", expand=False))
    
    while True:
        # 사용자로부터 질문을 입력받습니다.
        query = console.input("\n[bold yellow]질문을 입력하세요:[/bold yellow] ")
        if query.lower() == '종료':
            console.print("[bold red]프로그램을 종료합니다. 감사합니다![/bold red]")
            break
        
        try:
            # perplexity_style_search 함수를 호출하여 검색 및 응답 생성을 수행합니다.
            perplexity_style_search(query)
        except Exception as e:
            # 오류가 발생한 경우 오류 메시지를 출력합니다.
            console.print(f"[bold red]오류가 발생했습니다: {str(e)}[/bold red]")
        
        # 다음 질문을 위한 안내 메시지를 출력합니다.
        console.print("\n[italic]다음 질문을 입력하거나 '종료'를 입력하여 프로그램을 종료할 수 있습니다.[/italic]")

# 스크립트가 직접 실행될 때만 main 함수를 호출합니다.
if __name__ == "__main__":
    main()