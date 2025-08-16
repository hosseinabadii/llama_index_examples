import json

from llama_index.core.workflow import Context, JsonSerializer, Workflow


def save_context_to_file(file_path: str, ctx: Context):
    ctx_dict = ctx.to_dict(serializer=JsonSerializer())
    with open(file_path, "w") as f:
        json.dump(ctx_dict, f)


def load_context_from_file(file_path: str, workflow: Workflow):
    try:
        with open(file_path, "r") as f:
            ctx_dict = json.load(f)
            return Context.from_dict(workflow, ctx_dict, serializer=JsonSerializer())
    except FileNotFoundError:
        print("No context file found. Creating a new one.")
        return Context(workflow)
    except json.JSONDecodeError:
        print("Invalid JSON in context file. Creating a new one.")
        return Context(workflow)


def print_chat_history_from_context(ctx: Context):
    ctx_dict = ctx.to_dict(serializer=JsonSerializer())
    memory = ctx_dict["state"]["state_data"]["_data"]["memory"]
    chat_history = json.loads(memory)["value"]["chat_store"]["store"]["chat_history"]
    for message in chat_history:
        print(f"{message['role'].capitalize():10}: {message['blocks'][0]['text']}")
