#!/usr/bin/env python3
"""PixelRAG Agent — Claude + tool_use for visual web search.

A real Anthropic agent that uses Claude to answer questions by searching
a visual Wikipedia index via PixelRAG. Claude decides when to call the
search tool and synthesizes answers from visual retrieval results.

Prerequisites:
    - ANTHROPIC_API_KEY env var set
    - pixelrag serve running on localhost:30001 (or set --endpoint)
    - the Anthropic SDK: pip install anthropic

Usage:
    # Interactive conversation with the agent
    python demos/agent_skill.py

    # Single question
    python demos/agent_skill.py "Who invented the telephone?"

    # Custom endpoint
    python demos/agent_skill.py --endpoint http://gpu-box:30001 "Eiffel Tower history"
"""

import argparse
import json
import sys
import urllib.request

try:
    import anthropic
except ModuleNotFoundError:  # not a core dependency — see #197
    sys.exit("This demo needs the Anthropic SDK: pip install anthropic")

SEARCH_TOOL = {
    "name": "pixelrag_search",
    "description": (
        "Search a visual Wikipedia index using natural language queries. "
        "Returns ranked results with article URLs and relevance scores. "
        "Use this tool to find information about any topic — it searches "
        "screenshot-based embeddings of Wikipedia articles, so it works well "
        "for both textual and visual content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 20)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

WEB_FETCH_TOOL = {
    "name": "web_fetch",
    "description": (
        "Fetch the text content of a URL. Use this to read Wikipedia articles "
        "or other web pages found via search. Returns the page text content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
        },
        "required": ["url"],
    },
}

TOOLS = [SEARCH_TOOL, WEB_FETCH_TOOL]

SYSTEM_PROMPT = """\
You are a research assistant with access to a visual Wikipedia search engine (PixelRAG).
When asked a question, use the pixelrag_search tool to find relevant Wikipedia articles,
then synthesize an answer from the results. You may search multiple times with different
queries to gather comprehensive information. Cite your sources with Wikipedia URLs.

If search results are insufficient, say so honestly rather than guessing."""


def execute_pixelrag_search(
    query: str, n_results: int = 5, endpoint: str = "http://localhost:30001"
) -> dict:
    """Call the PixelRAG search API."""
    body = json.dumps(
        {"queries": [{"text": query}], "n_docs": min(n_results, 20)}
    ).encode()
    req = urllib.request.Request(
        f"{endpoint}/search",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    hits = data.get("results", [{}])[0].get("hits", [])
    results = []
    for hit in hits:
        url = hit.get("url", "")
        slug = url.split("/wiki/")[-1] if "/wiki/" in url else ""
        title = slug.replace("_", " ") if slug else url
        results.append(
            {
                "title": title,
                "url": url,
                "score": round(hit["score"], 4),
                "tile": f"tile_{hit.get('tile_index', '?')}_chunk_{hit.get('chunk_index', '?')}",
            }
        )
    return {"query": query, "results": results, "count": len(results)}


def execute_web_fetch(url: str) -> dict:
    """Fetch text from a URL (simplified — returns first 4000 chars)."""
    req = urllib.request.Request(url, headers={"User-Agent": "PixelRAG-Agent/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    # Strip HTML tags for a rough text extraction
    import re

    text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return {"url": url, "content": text[:4000], "truncated": len(text) > 4000}


def handle_tool_call(tool_name: str, tool_input: dict, endpoint: str) -> str:
    """Execute a tool call and return the result as a string."""
    try:
        if tool_name == "pixelrag_search":
            result = execute_pixelrag_search(
                query=tool_input["query"],
                n_results=tool_input.get("n_results", 5),
                endpoint=endpoint,
            )
        elif tool_name == "web_fetch":
            result = execute_web_fetch(url=tool_input["url"])
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result)


def run_agent(
    question: str,
    endpoint: str,
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = False,
) -> str:
    """Run the agent loop: send question → handle tool calls → return final answer."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if verbose:
            print(
                f"  [stop_reason={response.stop_reason}, usage={response.usage}]",
                file=sys.stderr,
            )

        if response.stop_reason == "end_turn":
            # Extract text from response
            text_parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_parts)

        # Handle tool use
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    print(
                        f"  [tool: {block.name}({json.dumps(block.input, ensure_ascii=False)})]",
                        file=sys.stderr,
                    )
                result = handle_tool_call(block.name, block.input, endpoint)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        if not tool_results:
            # No tool calls and not end_turn — shouldn't happen, but handle gracefully
            text_parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_parts) if text_parts else "(no response)"

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


def interactive(endpoint: str, model: str, verbose: bool):
    """Run interactive conversation loop."""
    print("PixelRAG Agent (Claude + visual search)")
    print(f"  endpoint: {endpoint}")
    print(f"  model:    {model}")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        print()
        try:
            answer = run_agent(question, endpoint, model, verbose)
            print(f"Agent: {answer}\n")
        except anthropic.APIError as e:
            print(f"API error: {e}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="PixelRAG Agent — Claude + visual web search"
    )
    parser.add_argument(
        "question", nargs="?", help="Question to ask (omit for interactive mode)"
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:30001",
        help="PixelRAG search API endpoint",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-20250514", help="Claude model to use"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show tool calls and API details"
    )
    args = parser.parse_args()

    if args.question:
        answer = run_agent(args.question, args.endpoint, args.model, args.verbose)
        print(answer)
    else:
        interactive(args.endpoint, args.model, args.verbose)


if __name__ == "__main__":
    main()
