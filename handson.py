"""Natural-language analysis with explicit, read-only pandas tools."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from langchain.tools import tool

DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "train.csv"


def build_tools(frame: pd.DataFrame) -> list[Any]:
    """Expose bounded calculations instead of executing model-generated Python."""

    @tool
    def inspect_data() -> str:
        """Return row/column counts, column types and missing-value counts."""
        return json.dumps({
            "rows": len(frame), "columns": len(frame.columns),
            "dtypes": frame.dtypes.astype(str).to_dict(),
            "missing": frame.isna().sum().to_dict(),
        }, ensure_ascii=False)

    @tool
    def summarize_numeric(column: str) -> str:
        """Describe a numeric column; null values are excluded and counted separately."""
        if column not in frame or not pd.api.types.is_numeric_dtype(frame[column]):
            return "오류: 실제 존재하는 숫자 열을 선택해 주세요."
        return frame[column].describe().to_json(force_ascii=False)

    @tool
    def compare_groups(group: str, value: str) -> str:
        """Return count, mean, median and sum by group, including null group keys. Maximum 50 groups."""
        if group not in frame or value not in frame:
            return "오류: 실제 존재하는 열을 선택해 주세요."
        if not pd.api.types.is_numeric_dtype(frame[value]):
            return "오류: 비교할 값은 숫자 열이어야 해요."
        if frame[group].nunique(dropna=False) > 50:
            return "오류: 그룹이 50개를 넘어요. 더 적은 범주의 열을 선택해 주세요."
        result = frame.groupby(group, dropna=False)[value].agg(["count", "mean", "median", "sum"])
        return result.to_json(orient="table", force_ascii=False)

    return [inspect_data, summarize_numeric, compare_groups]


def main() -> None:
    """Inspect locally or start an optional API-backed conversation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--check", action="store_true", help="로컬 데이터만 확인; API 호출 없음")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"CSV 파일을 찾을 수 없어요: {args.data}")
    if not args.check and not (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL")):
        parser.error("OPENAI_API_KEY와 도구 호출을 지원하는 OPENAI_MODEL을 설정해 주세요.")
    tools = build_tools(pd.read_csv(args.data))
    if args.check:
        print(tools[0].invoke({}))
        return

    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    agent = create_agent(
        model=ChatOpenAI(model=os.environ["OPENAI_MODEL"]),
        tools=tools,
        system_prompt=(
            "너는 통계 분석 학습 도우미다. 먼저 inspect_data로 열과 결측을 확인하라. "
            "숫자는 반드시 도구 결과로 확인하고 분석 단위·집계 방식·유효 관측 수를 밝혀라. "
            "식별자를 양적 변수로 해석하지 말고, 거래 행과 고유 고객 수를 구분하라. "
            "표본 차이를 원인이나 모집단 전체의 특성으로 단정하지 마라. "
            "지원하지 않는 계산·검정·그래프 요청은 미수행이라고 밝히고 필요한 다음 분석을 설명하라. "
            "한국어로 답하라."
        ),
    )
    messages: list[Any] = []
    while question := input("분석 질문 (Enter로 종료): ").strip():
        result = agent.invoke(
            {"messages": [*messages, {"role": "user", "content": question}]},
            config={"recursion_limit": 12},
        )
        new_messages = result["messages"][len(messages) + 1:]
        messages = result["messages"]
        for message in new_messages:
            if message.type == "tool":
                print(f"[계산 근거 · {message.name}] {message.content}")
        print(messages[-1].content)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n실습 종료")
