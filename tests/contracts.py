"""Independent, intentionally limited Responses consumer oracle.

This checks our documented translated-stream contract; it is NOT Muse's
decoder. Actual Muse acceptance is a separate command. No runtime converter,
serializer, or test expectations are imported from production code.
"""

import json


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def consume(wire):
    events = []
    text = wire.decode("utf-8").replace("\r\n", "\n")
    require(text.endswith("\n\n"), "SSE_FRAME: unclosed event")
    for block in text.split("\n\n"):
        if not block:
            continue
        lines = block.split("\n")
        names = [line[6:].strip() for line in lines if line.startswith("event:")]
        data = "\n".join(line[5:].lstrip(" ") for line in lines if line.startswith("data:"))
        if data == "[DONE]":
            continue  # Optional marker is not the completion signal.
        payload = json.loads(data)
        require(names == [payload.get("type")], "SSE_FRAME: event/data type mismatch")
        events.append(payload)
    require(bool(events), "TERMINAL: empty stream")
    numbers = [item.get("sequence_number") for item in events]
    require(all(type(n) is int for n in numbers) and numbers == list(range(len(events))),
            "STREAM_SEQUENCE: missing, duplicate, or out-of-order sequence_number")
    require(events[0]["type"] == "response.created", "LIFECYCLE: missing response.created")
    response_id = events[0]["response"]["id"]
    added, done, texts, arguments = {}, {}, {}, {}
    terminal = None
    for event in events:
        require(terminal is None, "TERMINAL: event after terminal")
        kind = event["type"]
        if "response" in event:
            require(event["response"]["id"] == response_id, "LIFECYCLE: response ID changed")
        if kind == "response.output_item.added":
            index = event["output_index"]
            require(index not in added, "ITEM: duplicate output index")
            added[index] = event["item"]
        elif "item_id" in event:
            index = event["output_index"]
            require(index in added and added[index]["id"] == event["item_id"],
                    "ITEM: delta refers to missing or different item")
            if kind == "response.output_text.delta":
                texts[index] = texts.get(index, "") + event["delta"]
            elif kind == "response.output_text.done":
                require(texts.get(index, "") == event["text"], "TEXT_REPLAY: delta/done mismatch")
            elif kind == "response.function_call_arguments.delta":
                arguments[index] = arguments.get(index, "") + event["delta"]
            elif kind == "response.function_call_arguments.done":
                require(arguments.get(index, "") == event["arguments"],
                        "TOOL_REPLAY: delta/done argument bytes differ")
        elif kind == "response.output_item.done":
            index, item = event["output_index"], event["item"]
            require(index in added and added[index]["id"] == item["id"], "ITEM: done ID changed")
            require(index not in done, "ITEM: duplicate done")
            done[index] = item
            if item["type"] == "function_call":
                require(bool(item["name"]) and bool(item["call_id"]), "TOOL_REPLAY: tool identity missing")
                require(arguments.get(index, "") == item["arguments"], "TOOL_REPLAY: final arguments differ")
                json.loads(item["arguments"])
            elif item["type"] == "message":
                expected = "".join(p["text"] for p in item["content"] if p["type"] == "output_text")
                require(texts.get(index, "") == expected, "TEXT_REPLAY: final text differs")
        if kind in {"response.completed", "response.failed", "response.incomplete"}:
            terminal = event["response"]
            require(terminal["status"] == kind.split(".")[1], "TERMINAL: status disagrees with event")
    require(terminal is not None, "TERMINAL: EOF without terminal event")
    if terminal["status"] != "failed":
        require(set(done) == set(added), "ITEM: unfinished output item")
        require(terminal["output"] == [done[index] for index in sorted(done)],
                "ITEM: terminal output disagrees with streamed items")
    return terminal
