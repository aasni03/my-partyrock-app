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
    "You are a biomedical knowledge extraction specialist. Carefully read the research paper below and "
    "identify all explicitly mentioned relationships between genes, diseases, and treatments or interventions. "
    "Present your findings as a markdown table with the following columns:\n\n"
    "| Gene | Disease | Treatment | Evidence |\n\n"
    "The Evidence column should contain a brief quote or description from the paper supporting the relationship. "
    "Only extract relationships clearly stated in the paper — do not infer or invent connections. "
    "If no gene-disease-treatment relationships are found in the paper, clearly state: "
    "\"No gene-disease-treatment relationships were identified in this paper.\""
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
