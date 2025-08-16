import asyncio
import logging

from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from tavily import AsyncTavilyClient

from utils import load_context_from_file, save_context_to_file

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

ctx = load_context_from_file("context.json", agent)


async def main():
    response = await agent.run(
        user_msg="tell me just the name of frameworks that you suggest in previous answer",
        ctx=ctx,
    )
    print(str(response))

    save_context_to_file("context.json", ctx)


if __name__ == "__main__":
    asyncio.run(main())
