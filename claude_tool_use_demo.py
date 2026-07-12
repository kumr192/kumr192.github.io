"""
Claude API concepts demo — for certification study.

This single script demonstrates, with heavy comments:

  1. SYSTEM message      — top-level `system` parameter (sets Claude's behavior)
  2. USER message        — {"role": "user", ...} in the `messages` array
  3. ASSISTANT message   — Claude's reply; you append it back to history yourself
  4. TOOL USE            — defining a tool, Claude requesting it (`tool_use` block),
                           you executing it, and returning a `tool_result`
  5. AGENT LOOP          — the while-loop that keeps calling the API until Claude
                           stops asking for tools
  6. STOP REASONS        — `tool_use` ("run a tool and call me back") vs
                           `end_turn` ("I'm done, here's my final answer")

Key mental model
----------------
The Messages API is STATELESS. Claude remembers nothing between HTTP calls.
Every request must carry the FULL conversation history in `messages`.
The "agent loop" is just: call API -> if Claude wants a tool, run it and
append the result to history -> call API again -> repeat until end_turn.

Message roles recap
-------------------
  system    : NOT in the messages array (normally) — it's the top-level
              `system` parameter. It sets persona/rules for the whole chat.
  user      : What the human said. ALSO used to carry tool_result blocks
              back to Claude (tool results ride in a user message!).
  assistant : What Claude said. Can contain plain text blocks AND tool_use
              blocks in the same message.

Stop reasons you should know for the cert
-----------------------------------------
  end_turn      : Claude finished its answer naturally.  -> exit the loop
  tool_use      : Claude wants you to run a tool.         -> run it, loop again
  max_tokens    : Output hit the max_tokens cap.          -> raise cap / stream
  stop_sequence : Hit a custom stop sequence you set.
  pause_turn    : Server-side tool loop paused; re-send to resume.
  refusal       : Claude declined for safety reasons.

Run it:
  pip install anthropic
  export ANTHROPIC_API_KEY=sk-ant-...
  python claude_tool_use_demo.py
"""

import json

import anthropic

# The client reads ANTHROPIC_API_KEY from the environment automatically.
client = anthropic.Anthropic()

MODEL = "claude-opus-4-8"

# ---------------------------------------------------------------------------
# 1) SYSTEM MESSAGE
#    Passed as the top-level `system` parameter, NOT inside `messages`.
#    It applies to the entire conversation and carries "operator authority".
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a concise weather assistant. "
    "Use the get_weather tool whenever the user asks about weather. "
    "Answer in one short sentence."
)

# ---------------------------------------------------------------------------
# 4) TOOL DEFINITION
#    A tool = name + description + JSON Schema for its inputs.
#    Claude reads the description to decide WHEN to call it, and the schema
#    to know HOW to shape the `input` it sends you.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city. "
            "Call this whenever the user asks about weather conditions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Paris' or 'Chennai'",
                },
            },
            "required": ["city"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> str:
    """YOUR code. The API never runs tools — it only ASKS you to run them.

    In a real app this would call a weather API. We fake it so the demo
    runs offline. Whatever string you return goes back to Claude as the
    tool_result content.
    """
    if name == "get_weather":
        # Pretend we looked it up.
        return f"Weather in {tool_input['city']}: 31°C, partly cloudy, humidity 78%."
    return f"Error: unknown tool '{name}'"


def main() -> None:
    # -----------------------------------------------------------------------
    # 2) USER MESSAGE — the conversation always starts with role "user".
    # -----------------------------------------------------------------------
    messages = [
        {"role": "user", "content": "What's the weather in Chennai right now?"},
    ]

    turn = 0

    # -----------------------------------------------------------------------
    # 5) THE AGENT LOOP
    #    Keep calling the API until stop_reason is no longer "tool_use".
    # -----------------------------------------------------------------------
    while True:
        turn += 1
        print(f"\n========== API call #{turn} ==========")

        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,   # <- system message goes here, not in messages
            tools=TOOLS,            # <- tool definitions ride on every request
            messages=messages,      # <- FULL history every time (API is stateless)
        )

        # 6) STOP REASON — why did Claude stop generating?
        print(f"stop_reason = {response.stop_reason!r}")

        # ------------------------------------------------------------------
        # 3) ASSISTANT MESSAGE
        #    Claude's response content is a LIST of blocks. During tool use
        #    it typically contains a text block ("Let me check...") followed
        #    by one or more tool_use blocks.
        #    We MUST append this assistant message to history — including the
        #    tool_use blocks — or the next request will be rejected.
        # ------------------------------------------------------------------
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"  [assistant text]     {block.text}")
            elif block.type == "tool_use":
                print(f"  [assistant tool_use] {block.name}({json.dumps(block.input)})  id={block.id}")

        # ------------------------------------------------------------------
        # EXIT CONDITION: "end_turn" means Claude gave its final answer.
        # ------------------------------------------------------------------
        if response.stop_reason == "end_turn":
            print("\nClaude is done — exiting the agent loop.")
            break

        # ------------------------------------------------------------------
        # "tool_use" means Claude paused and is waiting for tool results.
        # Execute EVERY tool_use block (Claude can request several in
        # parallel) and send ALL results back in a SINGLE user message.
        # Each tool_result must echo the matching tool_use block's `id`
        # as `tool_use_id` — that's how Claude pairs results to requests.
        # ------------------------------------------------------------------
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    print(f"  [we run the tool]    -> {result}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,  # must match block.id above
                            "content": result,
                            # On failure you'd add: "is_error": True
                        }
                    )

            # NOTE: tool results are sent back inside a USER message.
            messages.append({"role": "user", "content": tool_results})
            continue  # loop: call the API again with the updated history

        # Anything else (max_tokens, refusal, pause_turn...) — stop the demo.
        print(f"Unhandled stop_reason {response.stop_reason!r} — exiting.")
        break

    # -----------------------------------------------------------------------
    # Show the final conversation structure — this is the part worth studying.
    # Expected shape for a single tool call:
    #   1. user      : the question
    #   2. assistant : text + tool_use block          (stop_reason: tool_use)
    #   3. user      : tool_result block
    #   4. assistant : final text answer              (stop_reason: end_turn)
    # -----------------------------------------------------------------------
    print("\n========== Final message history (roles) ==========")
    for i, msg in enumerate(messages, 1):
        content = msg["content"]
        if isinstance(content, str):
            kinds = "text"
        else:
            kinds = ", ".join(
                b["type"] if isinstance(b, dict) else b.type for b in content
            )
        print(f"  {i}. {msg['role']:<9} [{kinds}]")


if __name__ == "__main__":
    main()
