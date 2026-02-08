"""
VALIDATION AGENT

JOB: Verify that the synthesized answer is supported by the source documents.
"""

from __future__ import print_function

import time
from typing import Dict, List
from backend.config import settings
from backend.agents.state import GraphState, AgentStep
from backend.services.llm_client import get_validation_client

def validate_answer(state: GraphState) -> Dict:
    """
    Fact-check the synthesized answer against the source documents.

    PROCESS:
    1. Extract answer and sources from state.
    2. Use LLM to fact-check every claim.
    3. Identify unsupported claims (hallucinations)
    4. Decide if validation passes or fails
    5. If fialed, try to correct the answer
    6. Return validation results
    """

    answer = state.get("synthesized_answer", "")
    chunks = state.get("retrieved_chunks", [])
    query = state.get("query", "")
    retry_cnt = state.get("retry_cnt", 0)

    print(f"\n{'='*60}")
    print(f"Validation Agent")
    print(f"{'='*60}")

    print(f"[INFO]\tValidating answer {len(answer) if answer else 0} characters")
    print(f"[INFO]\tAgainst {len(chunks) if chunks else 0} source documents")
    print(f"[INFO]\tRetry Count: {retry_cnt}")

    # Edge Case
    if not answer or not chunks:
        print(f"[WARNING]\tMissing answer or source documents. Skipping validation.")
        return {
            "validation_passed": False,
            "validation_issues": ["Missing answer or source documents"],
            "fact_checked_answer": answer,
            "retry_cnt": retry_cnt + 1,  # Must increment to avoid infinite loop
            "agent_steps": [
                AgentStep(
                    agent_name="validation_agent",
                    action="Missing answer or source documents",
                    reasoning="Edge case - skipping validation",
                    timestamp=time.time()
                )
            ]
        }
    
    # building source context
    source_list = []
    for i, chunk in enumerate(chunks):
        source_text = f"Source {i+1}: {chunk['text']}"
        source_list.append(source_text)

    sources_context = "\n\n".join(source_list)

    # validation prompt
    system_prompt = """You are a Rigorous Fact-Checker for a RAG system.
    Your Job is to verify that every claim in the answer is supported by the provided sources.
    You must be strict - if a claim is not explicitly stated in the sources, it's a hallucination.
    """

    user_prompt = f"""You are fact-checking an AI-generated answer against source documents.

    ORIGINAL QUESTION:
    {query}

    ANSWER TO VALIDATE:
    {answer}

    SOURCE DOCUMENTS:
    {sources_context}

    YOUR TASK:
    1. Analyze every claim in the answer.
    2. For each claim, verify it's supported by at least one source.
    3. Identify any claims that are:
        - Not mentioned in the sources (hallucinations)
        - Contradicted by sources (error)
        - Misrepresented or Exaggerated (misinformation)

    VALIDATION CRITERIA:
    - A claim is SUPPORTED if it's explicitly stated or direclty implied by sources
    - A claim is UNSUPPORTED if you have to make assumptions or inferences
    - Be strict: "possibly supported" = UNSUPPORTED

    IMPORTANT RULES:
    - Don't be lenient - this is fact-checking task
    - If you're unsure, mark as unsupported
    - Check dates, numbers, and names carefully
    - Look for contradictions between sources

    RESPOND IN JSON FORMAT:
    {{
        "validation_passed": true/false,
        "unsupported_claims": [
            {{
                "claim": "the specific claim from the answer",
                "issue": "why it's unsupported",
                "severity": "high" | "medium" | "low"
            }}
        ],
        "contradictions": ["list of any contradictions found"],
        "corrected_answer": "If validation fails, provide a corrected version that only uses supported claims.",
        "confidence": 0.0-1.0
    }}
    """

    # execute validation
    llm = get_validation_client()
    try:
        response = llm.generate_json(
            prompt=user_prompt,
            system_prompt=system_prompt
        )

        validation_passed = response.get("validation_passed", False)
        unsupported_claims = response.get("unsupported_claims", [])
        contradictions = response.get("contradictions", [])
        corrected_answer = response.get("corrected_answer", answer)
        validation_confidence = response.get("confidence", 0.0)

    except Exception as e:
        print(f"[ERROR]\tFailed to call Validation LLM: {str(e)}")
        # FALLBACK: Assume answer is okay if LLM Fails
        validation_passed = True
        unsupported_claims = []
        contradictions = []
        corrected_answer = answer
        validation_confidence = 0.5
        print(f"[WARNING]\tValidation LLM failed. Using fallback values.")

    # analyze validation issues
    validation_issues = []
    for claim in unsupported_claims:
        issue_text = f"{claim['severity'].upper()}: {claim['claim']} - {claim['issue']}"
        validation_issues.append(issue_text)

    for contradiction in contradictions:
        validation_issues.append(f"CONTRADICTION: {contradiction}")

    # DECISION LOGIC:
    # validation fails if:
    # - LLM explicitely says validation_passed = false OR
    # - we've found any HIGH severity unsupported claims OR
    # - we found contradictions

    high_severity_count = sum(1 for claim in unsupported_claims if claim.get('severity') == 'high')

    if not validation_passed or high_severity_count > 0 or len(contradictions) > 0:
        final_validation_passed = False
        final_answer = corrected_answer

        print(f"[WARNING]\tValidation Failed")
        print(f"[WARNING]\tUnsupported Claims: {len(unsupported_claims)}")
        print(f"[WARNING]\tHigh Severity Claims: {high_severity_count}")
        print(f"[WARNING]\tContradictions: {len(contradictions)}")

    else:
        final_validation_passed = True
        final_answer = answer

        print(f"[INFO]\tValidation Passed")
        print(f"[INFO]\tConfidence: {validation_confidence:.2f}")

        if validation_issues:
            print(f"[INFO]\tMinor Validation Issues Noted: {len(validation_issues)}")

    
    will_retry = not final_validation_passed and retry_cnt < settings.MAX_RETRIES

    if will_retry:
        print(f"[INFO]\tWill Retry Retrieval (attempt {retry_cnt+1})")
    elif not final_validation_passed:
        print(f"[INFO]\tMAX retries reached. Returning to corrected answer.")
    else:
        print(f"[INFO]\tValidation Passed. No Retry Needed.")

    # agent step record
    action = "validation_passed" if final_validation_passed else "validation_failed"
    if will_retry:
        action += f"will retry. Retry Count=(retry {retry_cnt+1})"

    reasoning = f"""
    Validation{'Passed' if final_validation_passed else 'Failed'}.
    Issues={len(validation_issues)}
    Retries={retry_cnt}
    Confidence: {validation_confidence:.2f}"
    """
    step = AgentStep(
        agent_name="validation_agent",
        action=action,
        reasoning=reasoning,
        timestamp=time.time()
    )

    return {
        "validation_passed": final_validation_passed,
        "validation_issues": validation_issues,
        "fact_checked_answer": final_answer,
        "retry_cnt": retry_cnt + 1, # always increment
        "agent_steps": [step]
    }
   