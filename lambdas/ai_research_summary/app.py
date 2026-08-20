import json
import base64
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0-20260217-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

SYSTEM_PROMPT = (
    "You are an expert biomedical research analyst. Using the research paper content provided below, "
    "generate a structured scientific summary with the following clearly labeled sections:\n\n"
    "**Research Title**\n"
    "**Research Objective**\n"
    "**Background**\n"
    "**Methodology**\n"
    "**Dataset Used**\n"
    "**Key Findings**\n"
    "**Conclusion**\n"
    "**Limitations**\n"
    "**Future Work**\n\n"
    "Use precise scientific language. Only extract information explicitly present in the paper. "
    "Do not invent or assume any facts. If a section cannot be determined from the provided text, "
    "state that it was not found in the paper."
)


def build_messages(body):
    paper_text = body.get("paper_text", "")
    file_data = body.get("file_data")
    file_mime = body.get("file_mime", "")

    content = []

    if file_data:
        if file_mime.startswith("image/"):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": file_mime,
                    "data": file_data,
                },
            })
        else:
            content.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": file_mime,
                    "data": file_data,
                },
            })

    text_payload = SYSTEM_PROMPT
    if paper_text:
        text_payload += f"\n\nResearch Paper:\n{paper_text}"
    elif not file_data:
        text_payload += "\n\nResearch Paper: (no content provided)"

    content.append({"type": "text", "text": text_payload})
    return [{"role": "user", "content": content}]


def generate(body):
    messages = build_messages(body)
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": messages,
    }
    response = bedrock.invoke_model_with_response_stream(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )
    for event in response["body"]:
        chunk = event.get("chunk")
        if chunk:
            data = json.loads(chunk["bytes"].decode("utf-8"))
            if data.get("type") == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    yield delta.get("text", "")


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@app.route("/", methods=["POST"])
def handler():
    body = request.get_json(force=True, silent=True) or {}

    def streamer():
        for chunk in generate(body):
            yield chunk

    resp = Response(stream_with_context(streamer()), content_type="text/plain; charset=utf-8")
    for k, v in CORS_HEADERS.items():
        resp.headers[k] = v
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
