"""
ANALYSIS AGENT

JOB: Read all Retrieved Documents and Write a Coherent answer.
"""

import time
from typing import Dict

from backend.agents.state import GraphState, AgentStep
from backend.services.llm_client import get_analysis_client

def analyze_and_synthesize(state: GraphState) -> Dict:
    """
    Synthesize a coherent answer from the retrieved documents.

    PROCESS:
    1. combine all documents into context
    2. create a synthesis prompt for LLM
    3. generate the answer
    4. Identify and gaps in the information
    5. Return the synthesized answer
    """

    query = state["query"]
    chunks = state.get("retrieved_chunks", [])

    print(f"\n{'='*60}")
    print(f"Analysis Agent")
    print(f"{'='*60}")

    print(f"[INFO]\tAnalyzing {len(chunks)} documents for query: {query[:50]}...")

    if not chunks or len(chunks) == 0:
        print(f"[WARNING]\tNo documents retrieved. Skipping analysis.")

        return {
            "synthesized_answer": "I couldn't find any relevant documents to answer your question. Pleae try rephrasing or asking something else.",
            "information_gaps": ["No documents retrieved"],
            "agent_steps": [
                AgentStep(
                    agent_name="analysis_agent",
                    action="No documents retrieved",
                    reasoning="No chunks available",
                    timestamp=time.time()
                )
            ]
        }

    print(f"[INFO]\tCombining {len(chunks)} documents into context...")

    # if there are chunks, then combine all into one text block
    context_parts = []

    for i, chunk in enumerate(chunks):
        header = f"[Source {i+1} (Score: {chunk['score']:.2f})]"

        if "title" in chunk["metadata"]:
            header += f" - {chunk['metadata']['title']}"
        
        context_parts.append(f"{header}\n{chunk['text']}")

    # join all with seperators
    separator = "\n" + "="*60 + "\n"
    context = separator.join(context_parts)

    print(f"[INFO]\tContext Length: {len(context)} characters")

    # create synthesis prompt
    system_prompt = """You are an Expert Technical Writer and Researcher.
    Your job is to synthesize information from multiple sources into clear, accurate answers.
    Adopt a Chain-of-Thought (CoT) approach: analyze the question, evaluate the sources, and plan your response before writing."""

    user_prompt = f"""You are answering a user's question using provided source documents.

    USER QUESTION:
    {query}

    SOURCE DOCUMENTS:
    {context}

    INSTRUCTION:
    1. ANALYSIS: Briefly analyze what the user is asking.
    2. EVALUATION: Check which sources contain relevant information.
    3. SYNTHESIS: Combine information to answer the question.
    4. CITATION: Cite your sources for every key claim (e.g., [Source 1]).

    IMPORTANT RULES:
    - DO NOT make up information not in the sources.
    - DO NOT add your own knowledge beyond the sources.
    - If sources conflict, mention both perspectives.
    - If sources don't fully answer the question, note what's missing.

    RESPOND in JSON FORMAT:
    {{
        "thought_process": "Brief analysis of the question and plan",
        "answer": "Your comprehensive answer here",
        "information_gaps": ["list", "of", "missing", "information"],
        "confidence": 0.0-1.0
    }}"""

    llm = get_analysis_client()

    try:
        response = llm.generate_json(
            prompt=user_prompt,
            system_prompt=system_prompt
        )

        synthesized_answer = response.get("answer", "")
        information_gaps = response.get("information_gaps", [])
        synthesized_confidence = response.get("confidence", 0.0)

        print(f"[INFO]\tSynthesized Answer: {synthesized_answer[:100]}...")
        print(f"[INFO]\tInformation Gaps: {information_gaps}")
        print(f"[INFO]\tConfidence: {synthesized_confidence:.2f}")

    except Exception as e:
        print(f"[ERROR]\tFailed to synthesize answer: {str(e)}")
        synthesized_answer = "I encountered an error while synthesizing your answer. Please try again later."
        information_gaps = ["Error in synthesis process"]
        synthesized_confidence = 0.0

    # Quality Check
    if len(synthesized_answer) < 50:
        print(f"[WARNING]\tSynthesized answer is too short.")
        information_gaps.append("Answer may be incomplete")

    # answer keywords should appear in the answer
    query_words = set(query.lower().split())
    answer_words = set(synthesized_answer.lower().split())
    overlap = len(query_words & answer_words) / len(query_words) if query_words else 0

    if overlap < 0.3: # less than 30# of word overlap
        print(f"[WARNING]\tLow keyword overlap ({overlap:.2f}). Answer may not be relevant.")
        information_gaps.append("Answer may not be relevant")

    step = AgentStep(
        agent_name="analysis_agent",
        action=f"synthesized answer from {len(chunks)} sources",
        reasoning=f"confidence={synthesized_confidence:.2f} gaps={len(information_gaps)}, Keyword overlap: {overlap:.2f}",
        timestamp=time.time()
    )

    # return the result
    return {
        "synthesized_answer": synthesized_answer,
        "information_gaps": information_gaps,
        "agent_steps": [step]
    }
    