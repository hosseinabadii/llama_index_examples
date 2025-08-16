import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Optional

from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

logging.basicConfig(level=logging.INFO)

load_dotenv()

llm = OpenAI(model="gpt-4o-mini")


async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient()
    return str(await client.search(query))


research_agent = FunctionAgent(
    name="ResearchAgent",
    description="Useful for searching the web for information on a given topic and recording notes on the topic.",
    system_prompt=(
        "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
        "You should output notes on the topic in a structured format."
    ),
    llm=llm,
    tools=[search_web],
)

write_agent = FunctionAgent(
    name="WriteAgent",
    description="Useful for writing a report based on the research notes or revising the report based on feedback.",
    system_prompt=(
        "You are the WriteAgent that can write a report on a given topic. "
        "Your report should be in a markdown format. The content should be grounded in the research notes. "
        "Return your markdown report surrounded by <report>...</report> tags."
    ),
    llm=llm,
)

review_agent = FunctionAgent(
    name="ReviewAgent",
    description="Useful for reviewing a report and providing feedback.",
    system_prompt=(
        "You are the ReviewAgent that can review the write report and provide feedback. "
        "Your review should either approve the current report or request changes for the WriteAgent to implement. "
    ),
    llm=llm,
)


# Write helper functions to help execute each agent


async def call_research_agent(ctx: Context, prompt: str) -> str:
    """Useful for recording research notes based on a specific prompt."""
    result = await research_agent.run(user_msg=f"Write some notes about the following: {prompt}")

    async with ctx.store.edit_state() as ctx_state:
        ctx_state["state"]["research_notes"].append(str(result))

    return str(result)


async def call_write_agent(ctx: Context) -> str:
    """Useful for writing a report based on the research notes or revising the report based on feedback."""
    async with ctx.store.edit_state() as ctx_state:
        notes = ctx_state["state"].get("research_notes", None)
        if not notes:
            return "No research notes to write from."

        user_msg = "Write a markdown report from the following notes. Be sure to output the report in the following format: ...:\n\n"

        # Add the feedback to the user message if it exists
        feedback = ctx_state["state"].get("review", None)
        if feedback:
            user_msg += f"{feedback}\n\n"

        # Add the research notes to the user message
        notes = "\n\n".join(notes)
        user_msg += f"{notes}\n\n"

        # Run the write agent
        result = await write_agent.run(user_msg=user_msg)
        report = re.search(r"(.*)", str(result), re.DOTALL).group(1)  # type: ignore
        ctx_state["state"]["report_content"] = str(report)

    return str(report)


async def call_review_agent(ctx: Context) -> str:
    """Useful for reviewing the report and providing feedback."""
    async with ctx.store.edit_state() as ctx_state:
        report = ctx_state["state"].get("report_content", None)
        if not report:
            return "No report content to review."

        result = await review_agent.run(user_msg=f"Review the following report: {report}")
        ctx_state["state"]["review"] = result

    return result


# Defining the Planner Workflow

PLANNER_PROMPT = """You are a planner chatbot. 

Given a user request and the current state, break the solution into ordered <step> blocks.  Each step must specify the agent to call and the message to send, e.g.
<plan>
  <step agent=\"ResearchAgent\">search for …</step>
  <step agent=\"WriteAgent\">draft a report …</step>
  ...
</plan>

<state>
{state}
</state>

<available_agents>
{available_agents}
</available_agents>

The general flow should be:
- Record research notes
- Write a report
- Review the report
- Write the report again if the review is not positive enough

If the user request does not require any steps, you can skip the <plan> block and respond directly.
"""


class InputEvent(StartEvent):
    user_msg: Optional[str] = Field(default=None)
    chat_history: list[ChatMessage]
    state: Optional[dict[str, Any]] = Field(default=None)


class OutputEvent(StopEvent):
    response: str
    chat_history: list[ChatMessage]
    state: dict[str, Any]


class StreamEvent(Event):
    delta: str


class PlanEvent(Event):
    step_info: str


# Modelling the plan
class PlanStep(BaseModel):
    agent_name: str
    agent_input: str


class Plan(BaseModel):
    steps: list[PlanStep]


class ExecuteEvent(Event):
    plan: Plan
    chat_history: list[ChatMessage]


class PlannerWorkflow(Workflow):
    llm: OpenAI = OpenAI(model="o3-mini")
    agents: dict[str, FunctionAgent] = {
        "ResearchAgent": research_agent,
        "WriteAgent": write_agent,
        "ReviewAgent": review_agent,
    }

    @step
    async def plan(self, ctx: Context, ev: InputEvent) -> ExecuteEvent | OutputEvent:
        # Set initial state if it exists
        if ev.state:
            await ctx.store.set("state", ev.state)

        chat_history = ev.chat_history

        if ev.user_msg:
            user_msg = ChatMessage(role="user", content=ev.user_msg)
            chat_history.append(user_msg)

        # Inject the system prompt with state and available agents
        state = await ctx.store.get("state")
        available_agents_str = "\n".join([f"{agent.description}" for agent in self.agents.values()])
        system_prompt = ChatMessage(
            role="system",
            content=PLANNER_PROMPT.format(
                state=str(state),
                available_agents=available_agents_str,
            ),
        )

        # Stream the response from the llm
        response = await self.llm.astream_chat(
            messages=[system_prompt] + chat_history,
        )
        full_response = ""
        async for chunk in response:
            full_response += chunk.delta or ""
            if chunk.delta:
                ctx.write_event_to_stream(
                    StreamEvent(delta=chunk.delta),
                )

        # Parse the response into a plan and decide whether to execute or output
        xml_match = re.search(r"(.*)", full_response, re.DOTALL)

        if not xml_match:
            chat_history.append(
                ChatMessage(
                    role="assistant",
                    content=full_response,
                )
            )
            return OutputEvent(
                response=full_response,
                chat_history=chat_history,
                state=state,
            )
        else:
            xml_str = xml_match.group(1)
            root = ET.fromstring(xml_str)
            plan = Plan(steps=[])
            for plan_step in root.findall("step"):
                plan.steps.append(
                    PlanStep(
                        agent_name=plan_step.attrib["agent"],
                        agent_input=plan_step.text.strip() if plan_step.text else "",
                    )
                )

            return ExecuteEvent(plan=plan, chat_history=chat_history)

    @step
    async def execute(self, ctx: Context, ev: ExecuteEvent) -> InputEvent:
        chat_history = ev.chat_history
        plan = ev.plan

        for plan_step in plan.steps:
            _ = self.agents[plan_step.agent_name]
            agent_input = plan_step.agent_input
            ctx.write_event_to_stream(
                PlanEvent(step_info=f"{plan_step.agent_input}"),
            )

            if plan_step.agent_name == "ResearchAgent":
                await call_research_agent(ctx, agent_input)
            elif plan_step.agent_name == "WriteAgent":
                # Note: we aren't passing the input from the plan since
                # we're using the state to drive the write agent
                await call_write_agent(ctx)
            elif plan_step.agent_name == "ReviewAgent":
                await call_review_agent(ctx)

        state = await ctx.store.get("state")
        chat_history.append(
            ChatMessage(
                role="user",
                content=f"I've completed the previous steps, here's the updated state:\n\n\n{state}\n\n\nDo you need to continue and plan more steps?, If not, write a final response.",
            )
        )

        return InputEvent(
            chat_history=chat_history,
        )


async def main() -> None:
    planner_workflow = PlannerWorkflow(timeout=None)

    handler = planner_workflow.run(
        user_msg=(
            "Write me a report on the history of the internet. "
            "Briefly describe the history of the internet, including the development of the internet, the development of the web, "
            "and the development of the internet in the 21st century."
        ),
        chat_history=[],
        state={
            "research_notes": [],
            "report_content": "Not written yet.",
            "review": "Review required.",
        },
    )

    async for event in handler.stream_events():
        if isinstance(event, PlanEvent):
            print("Executing plan step: ", event.step_info)
        elif isinstance(event, ExecuteEvent):
            print("Executing plan: ", event.plan)

    state = await handler.ctx.store.get("state")  # type: ignore

    print("\n---Report content result:")
    print(state["report_content"])

    print("\n---Review...")
    print(state["review"])


if __name__ == "__main__":
    asyncio.run(main())
