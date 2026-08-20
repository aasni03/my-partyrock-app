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
    "You are a senior biomedical research reviewer with expertise in comparative literature analysis. "
    "Compare the two research papers provided below. Structure your comparison with the following clearly "
    "labeled sections:\n\n"
    "**Similarities**: Common themes in methodology, findings, and conclusions\n"
    "**Differences**: Contrasts in research approach, results, and interpretations\n"
    "**Strength of Evidence**: Assess which paper presents stronger scientific evidence and explain why, "
    "considering factors like study design, sample size, statistical rigor, and reproducibility\n"
    "**Overall Assessment**: A brief synthesis of what the two papers together contribute to the field\n\n"
    "If only one paper has been provided and the second paper field is empty or missing, clearly state: "
    "\"A second paper is required for comparison. Please upload a second research paper using the "
    "Second Paper Upload input.\""
)


def build_messages(body):
    paper_text = body.get("paper_text", "")
    file_data = body.get("file_data")
    file_mime = body.get("file_mime", "")
    paper2_text = body.get("paper2_text", "")
    file2_data = body.get("file2_data")
    file2_mime = body.get("file2_mime", "")

    content = []

    # First paper
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

    # Second paper
    if file2_data:
        if file2_mime.startswith("image/"):
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": file2_mime, "data": file2_data},
            })
        else:
            content.append({
                "type": "document",
                "source": {"type": "base64", "media_type": file2_mime, "data": file2_data},
            })

    text_payload = SYSTEM_PROMPT
    text_payload += f"\n\nFirst Paper:\n{paper_text if paper_text else '(no text content — see uploaded document above)'}"
    text_payload += f"\n\nSecond Paper:\n{paper2_text if paper2_text else ('(no text content — see uploaded document above)' if file2_data else '(not provided)')}"

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
