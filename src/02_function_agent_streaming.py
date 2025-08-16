import asyncio
import logging

from dotenv import load_dotenv
from llama_index.core.agent.workflow import (
    AgentInput,  # noqa: F401
    AgentOutput,  # noqa: F401
    AgentStream,
    FunctionAgent,
    ToolCall,  # noqa: F401
    ToolCallResult,  # noqa: F401
)
from llama_index.llms.openai import OpenAI
from tavily import AsyncTavilyClient

logging.basicConfig(level=logging.INFO)
load_dotenv()

llm = OpenAI(model="gpt-4o-mini")


async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient()
    return str(await client.search(query))


agent = FunctionAgent(
    tools=[search_web],
    llm=llm,
    system_prompt="You are a helpful assistant that can search the web for information.",
)


async def main():
    handler = agent.run(user_msg="What is the weather in Saskatoon?")
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)
            # print(event.response)  # the current full response
            # print(event.raw)  # the raw llm api response
            # print(event.current_agent_name)  # the current agent name
        # elif isinstance(event, AgentInput):
        #     print(event.input)  # the current input messages
        #     print(event.current_agent_name)  # the current agent name
        # elif isinstance(event, AgentOutput):
        #     print(event.response)  # the current full response
        #     print(event.tool_calls)  # the selected tool calls, if any
        #     print(event.raw)  # the raw llm api response
        # elif isinstance(event, ToolCallResult):
        #     print(event.tool_name)  # the tool name
        #     print(event.tool_kwargs)  # the tool kwargs
        #     print(event.tool_output)  # the tool output
        # elif isinstance(event, ToolCall):
        #     print(event.tool_name)  # the tool name
        #     print(event.tool_kwargs)  # the tool kwargs


if __name__ == "__main__":
    asyncio.run(main())
