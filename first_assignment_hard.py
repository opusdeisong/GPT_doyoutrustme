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

# Rich 콘솔 초기화
console = Console()

# 환경 변수 로드
load_dotenv()

# API 키 설정
serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not serper_api_key or not openai_api_key:
    raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# 검색 도구 및 LLM 초기화
search = GoogleSerperAPIWrapper()
llm = ChatOpenAI(temperature=0)

# 검색 및 분석 체인
search_template = """
다음 질문 및 키워드에 대한 검색 결과를 분석하고 요약해주세요:
질문: {query}
키워드: {keywords}

검색 결과:
{search_result}

요약:
"""
search_prompt = ChatPromptTemplate.from_template(search_template)

search_chain = (
    {"query": RunnablePassthrough(), "keywords": RunnablePassthrough(), "search_result": lambda x: search.run(f"{x['query']} {x['keywords']}")}
    | search_prompt
    | llm
    | StrOutputParser()
)

# 추가 검색 필요성 및 키워드 판단 체인
need_more_search_template = """
다음 정보를 바탕으로, 추가 검색이 필요한지 판단하고 필요하다면 검색 키워드를 제시해주세요:

질문: {query}
현재까지의 분석: {current_analysis}

형식:
필요여부: [Yes/No]
키워드: [추가 검색이 필요한 경우 3개 이하의 키워드, 불필요한 경우 "없음"]

결정:
"""
need_more_search_prompt = ChatPromptTemplate.from_template(need_more_search_template)

need_more_search_chain = need_more_search_prompt | llm | StrOutputParser()

# 최종 응답 생성 체인
final_answer_template = """
다음 정보를 바탕으로 최종 응답을 생성해주세요:

질문: {query}
전체 분석 내용: {full_analysis}

최종 응답:
"""
final_answer_prompt = ChatPromptTemplate.from_template(final_answer_template)

final_answer_chain = final_answer_prompt | llm | StrOutputParser()

def parse_decision(decision):
    lines = decision.split('\n')
    need_more = lines[0].split(':')[1].strip().lower() == 'yes'
    keywords = lines[1].split(':')[1].strip()
    return need_more, keywords

def perplexity_style_search(query):
    console.print(Panel(f"[bold cyan]질문:[/bold cyan] {query}", expand=False))

    full_analysis = ""
    cumulative_keywords = ""
    iteration = 0
    max_iterations = 3

    while iteration < max_iterations:
        # 검색 및 분석 수행
        with console.status(f"[bold green]검색 및 분석 중... (반복 {iteration + 1}/{max_iterations})[/bold green]"):
            current_analysis = search_chain.invoke({"query": query, "keywords": cumulative_keywords})
        
        console.print(Panel(Text(f"분석 결과 (반복 {iteration + 1}/{max_iterations}):\n{current_analysis}", style="green"), expand=False))
        
        full_analysis += f"\n반복 {iteration + 1} 분석:\n{current_analysis}\n"

        # 추가 검색 필요성 판단
        with console.status("[bold yellow]추가 검색 필요성 판단 중...[/bold yellow]"):
            decision = need_more_search_chain.invoke({"query": query, "current_analysis": full_analysis})
        
        need_more, new_keywords = parse_decision(decision)
        console.print(f"[bold magenta]추가 검색 필요:[/bold magenta] {'예' if need_more else '아니오'}")
        console.print(f"[bold magenta]추가 검색 키워드:[/bold magenta] {new_keywords}")

        if not need_more or new_keywords.lower() == "없음":
            break

        # 새 키워드를 누적 키워드에 추가
        if cumulative_keywords:
            cumulative_keywords += ", " + new_keywords
        else:
            cumulative_keywords = new_keywords

        console.print(f"[bold blue]누적 검색 키워드:[/bold blue] {cumulative_keywords}")

        iteration += 1
    # 최종 응답 생성
    with console.status("[bold red]최종 응답 생성 중...[/bold red]"):
        final_answer = final_answer_chain.invoke({
            "query": query,
            "full_analysis": full_analysis
        })

    console.print(Panel(Text(f"최종 응답:\n{final_answer}", style="bold white on blue"), expand=False))

    return final_answer

# 메인 루프
def main():
    console.print(Panel("[bold green]누적 검색 결과를 활용하는 AI 검색 시스템에 오신 것을 환영합니다![/bold green]\n'종료'를 입력하면 프로그램이 종료됩니다.", expand=False))
    
    while True:
        query = console.input("\n[bold yellow]질문을 입력하세요:[/bold yellow] ")
        if query.lower() == '종료':
            console.print("[bold red]프로그램을 종료합니다. 감사합니다![/bold red]")
            break
        
        try:
            perplexity_style_search(query)
        except Exception as e:
            console.print(f"[bold red]오류가 발생했습니다: {str(e)}[/bold red]")
        
        console.print("\n[italic]다음 질문을 입력하거나 '종료'를 입력하여 프로그램을 종료할 수 있습니다.[/italic]")

if __name__ == "__main__":
    main()