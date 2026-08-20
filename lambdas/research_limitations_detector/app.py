import json
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
    "You are a rigorous peer reviewer with expertise in biomedical research methodology. Critically analyze "
    "the research paper below and identify potential weaknesses, biases, and methodological limitations in "
    "the study. Present your findings as a bulleted list. For each limitation, provide a brief explanation "
    "of why it is a concern and how it might affect the validity or generalizability of the findings. "
    "Consider sample sizes, controls, statistical methods, potential confounders, reproducibility, and "
    "scope of conclusions. Distinguish between limitations the authors themselves acknowledge and those "
    "you identify independently."
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
                "source": {"type": "base64", "media_type": file_mime, "data": file_data},
            })
        else:
            content.append({
                "type": "document",
                "source": {"type": "base64", "media_type": file_mime, "data": file_data},
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
        "max_tokens": 3000,
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
