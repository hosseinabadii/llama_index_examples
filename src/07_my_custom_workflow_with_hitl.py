import asyncio
import string
from pathlib import Path

from llama_index.core.workflow import (
    Context,
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEPARATOR_OPTIONS = {
    "1": " | ",
    "2": " : ",
    "3": " ; ",
    "4": " , ",
    "5": " - ",
}


INPUT_REQUIRED_PREFIX = f"""
Do you accept the final report? enter 'yes' to confirm.
OR
Select a separator:
1. {SEPARATOR_OPTIONS["1"]}
2. {SEPARATOR_OPTIONS["2"]}
3. {SEPARATOR_OPTIONS["3"]}
4. {SEPARATOR_OPTIONS["4"]}
5. {SEPARATOR_OPTIONS["5"]}

Your choice: """


def read_file(file_path: str) -> str:
    """Read a file and return its contents"""
    raw_text = ""
    try:
        with open(file_path) as f:
            raw_text = f.read()
    except FileNotFoundError as e:
        print(f"Error: {e}")

    return raw_text


async def clean_text(ctx: Context, text: str) -> str:
    """Clean text by removing punctuation and digits"""
    ctx.write_event_to_stream(
        InputRequiredEvent(
            prefix="\nUse 'clean_text' tool? (yes or no): ",
            user_name="Laurie",
        )
    )

    response = await ctx.wait_for_event(HumanResponseEvent, requirements={"user_name": "Laurie"})
    if response.response.lower() != "yes":
        return text

    return "".join(c for c in text.strip() if c not in string.punctuation + string.digits)


async def extract_keywords(ctx: Context, text: str) -> list[str]:
    """Extract keywords from text"""

    ctx.write_event_to_stream(
        InputRequiredEvent(
            prefix="\nUse 'extract_keywords' tool? (yes or no): ",
            user_name="Laurie",
        )
    )

    response = await ctx.wait_for_event(HumanResponseEvent, requirements={"user_name": "Laurie"})
    if response.response.lower() != "yes":
        return []
    words = text.split()

    return [word for word in words if len(word) > 4]


async def generate_report(ctx: Context, keywords: list[str], separator: str = " - ") -> str:
    """Generate a report based on keywords"""

    return f"Keywords: {separator.join(keywords)}"


class LoadDocumentEvent(StartEvent):
    file_path: str


class TextCleanEvent(Event):
    raw_text: str


class KeywordExtractEvent(Event):
    cleaned_text: str


class ReportGenerateEvent(Event):
    keywords: list[str]
    separator: str = " - "


class ConfirmReportEvent(Event):
    final_report: str


class MyWorkflow(Workflow):
    @step
    async def load_document_step(self, ctx: Context, ev: LoadDocumentEvent) -> TextCleanEvent | StopEvent:
        raw_text = read_file(ev.file_path)
        if raw_text == "":
            return StopEvent(result="No text found in file.")

        return TextCleanEvent(raw_text=raw_text)

    @step
    async def clean_text_step(self, ctx: Context, ev: TextCleanEvent) -> KeywordExtractEvent:
        cleaned_text = await clean_text(ctx, ev.raw_text)

        return KeywordExtractEvent(cleaned_text=cleaned_text)

    @step
    async def extract_keywords(self, ctx: Context, ev: KeywordExtractEvent) -> ReportGenerateEvent:
        keywords = await extract_keywords(ctx, ev.cleaned_text)
        async with ctx.store.edit_state() as ctx_state:
            ctx_state["keywords"] = keywords

        return ReportGenerateEvent(keywords=keywords)

    @step
    async def generate_report(self, ctx: Context, ev: ReportGenerateEvent) -> ConfirmReportEvent:
        final_report = await generate_report(ctx, ev.keywords, ev.separator)
        confirm_report_event = ConfirmReportEvent(final_report=final_report)
        ctx.write_event_to_stream(confirm_report_event)
        return confirm_report_event

    @step
    async def confirm_report(self, ctx: Context, ev: ConfirmReportEvent) -> ReportGenerateEvent | StopEvent:
        ctx.write_event_to_stream(
            InputRequiredEvent(
                prefix=INPUT_REQUIRED_PREFIX,
                user_name="Laurie",
            )
        )

        response = await ctx.wait_for_event(HumanResponseEvent, requirements={"user_name": "Laurie"})
        answer = response.response.lower()
        if answer == "yes":
            return StopEvent(result=ev.final_report)
        elif answer in SEPARATOR_OPTIONS:
            keywords = await ctx.store.get("keywords")
            return ReportGenerateEvent(keywords=keywords, separator=SEPARATOR_OPTIONS[answer])
        else:
            return StopEvent(result="Your input is not valid")


async def main():
    workflow = MyWorkflow(timeout=None)
    handler = workflow.run(
        file_path=str(DATA_DIR / "file.txt"),
    )

    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            response = input(event.prefix)
            handler.ctx.send_event(
                HumanResponseEvent(
                    response=response,
                    user_name=event.user_name,
                )
            )
        elif isinstance(event, ConfirmReportEvent):
            print(f"\n## Final Report:\n{event.final_report}")

    response = await handler
    print(f"\n## Final report:\n{response}")


if __name__ == "__main__":
    asyncio.run(main())
