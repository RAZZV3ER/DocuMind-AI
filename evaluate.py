
import os
import json
import time
from dotenv import load_dotenv
from groq import Groq
from agent_engine import run_agent

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")

client = Groq(api_key=API_KEY)
MODEL = "openai/gpt-oss-20b"


# --------------------------------
# EVALUATION DOCUMENT
# --------------------------------

DOCUMENT = """
Employee Name: Rahul Sharma
Annual CTC: Rs. 600000
Monthly Salary: Rs. 50000
Joining Date: 1 July 2026
Probation Period: 6 months
"""


# --------------------------------
# TEST CASES
# --------------------------------

TEST_CASES = [
    {
        "name": "Annual CTC retrieval",
        "task": (
            "Search the document using search_document to find "
            "the annual CTC. Report the amount."
        ),
        "expected": ["600000", "600,000"],
        "required_tools": ["search_document"],
    },
    {
        "name": "Probation period retrieval",
        "task": (
            "Search the document using search_document to find "
            "the probation period. Report its duration in months."
        ),
        "expected": ["6 months", "6 month"],
        "required_tools": ["search_document"],
    },
    {
        "name": "Monthly salary calculation",
        "task": (
            "Use search_document to find the annual CTC. "
            "Then use calculate to divide the annual CTC by 12. "
            "Report the calculation and monthly amount."
        ),
        "expected": ["50000", "50,000"],
        "required_tools": ["search_document", "calculate"],
    },
    {
        "name": "Missing information",
        "task": (
            "Use search_document to search for the notice period. "
            "If the document does not contain it, clearly say "
            "that the notice period was not found. Do not guess."
        ),
        "expected": [
            "not found",
            "not mentioned",
            "not specified",
            "does not mention",
            "does not contain",
            "no mention",
            "no information",
        ],
        "required_tools": ["search_document"],
    },
]


# --------------------------------
# ANSWER CHECK
# --------------------------------

def check_answer(answer, expected_phrases):
    """Check whether at least one expected phrase appears."""

    answer_lower = answer.lower()

    return any(
        phrase.lower() in answer_lower
        for phrase in expected_phrases
    )


# --------------------------------
# EVALUATION
# --------------------------------

def evaluate():
    results = []

    for index, test in enumerate(TEST_CASES, start=1):
        print(f"\nRunning test {index}: {test['name']}")

        start_time = time.perf_counter()

        try:
            result = run_agent(
                task=test["task"],
                document_text=DOCUMENT,
                groq_client=client,
                model=MODEL,
            )

            elapsed = time.perf_counter() - start_time

            answer = result.get("answer") or ""
            trace = result.get("trace") or []

            # Check expected answer content.
            answer_correct = check_answer(
                answer,
                test["expected"],
            )

            # Ignore the planning entry; inspect actual tool executions.
            tools_used = [
                item.get("tool")
                for item in trace
                if item.get("tool") not in (None, "plan")
            ]

            # Every required tool must actually appear in the trace.
            missing_tools = [
                tool
                for tool in test["required_tools"]
                if tool not in tools_used
            ]

            required_tools_used = len(missing_tools) == 0

            row = {
                "test": test["name"],
                "completed": True,
                "answer_check": answer_correct,
                "required_tools_check": required_tools_used,
                "required_tools": test["required_tools"],
                "missing_tools": missing_tools,
                "tools_used": tools_used,
                "latency_seconds": round(elapsed, 2),
                "answer": answer,
            }

        except Exception as error:
            elapsed = time.perf_counter() - start_time

            row = {
                "test": test["name"],
                "completed": False,
                "answer_check": False,
                "required_tools_check": False,
                "required_tools": test["required_tools"],
                "missing_tools": test["required_tools"],
                "tools_used": [],
                "latency_seconds": round(elapsed, 2),
                "answer": f"ERROR: {error}",
            }

        results.append(row)

        print("Answer check:", row["answer_check"])
        print("Required tools check:", row["required_tools_check"])
        print("Required tools:", row["required_tools"])
        print("Tools used:", row["tools_used"])
        print("Missing tools:", row["missing_tools"])
        print("Latency:", row["latency_seconds"], "seconds")
        print("Answer:", row["answer"])

    # --------------------------------
    # SUMMARY METRICS
    # --------------------------------

    total = len(results)

    completed_tests = sum(
        row["completed"] for row in results
    )

    answer_passes = sum(
        row["answer_check"] for row in results
    )

    tool_passes = sum(
        row["required_tools_check"] for row in results
    )

    # A test passes end-to-end only if both checks pass.
    end_to_end_passes = sum(
        row["completed"]
        and row["answer_check"]
        and row["required_tools_check"]
        for row in results
    )

    latencies = [
        row["latency_seconds"]
        for row in results
        if row["completed"]
    ]

    average_latency = (
        sum(latencies) / len(latencies)
        if latencies else 0
    )

    summary = {
        "total_tests": total,
        "completed_tests": completed_tests,
        "answer_keyword_pass_rate_percent": round(
            answer_passes / total * 100, 2
        ) if total else 0,
        "required_tool_use_rate_percent": round(
            tool_passes / total * 100, 2
        ) if total else 0,
        "end_to_end_pass_rate_percent": round(
            end_to_end_passes / total * 100, 2
        ) if total else 0,
        "average_latency_seconds": round(
            average_latency, 2
        ),
        "results": results,
    }

    # Save the detailed evaluation report.
    with open(
        "evaluation_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n========== EVALUATION SUMMARY ==========")
    print("Total tests:", total)
    print("Completed tests:", completed_tests)

    print(
        "Answer keyword pass rate:",
        summary["answer_keyword_pass_rate_percent"], "%",
    )

    print(
        "Required tool use rate:",
        summary["required_tool_use_rate_percent"], "%",
    )

    print(
        "End-to-end pass rate:",
        summary["end_to_end_pass_rate_percent"], "%",
    )

    print(
        "Average latency:",
        summary["average_latency_seconds"], "seconds",
    )

    print("\nDetailed results saved to evaluation_results.json")


if __name__ == "__main__":
    evaluate()
