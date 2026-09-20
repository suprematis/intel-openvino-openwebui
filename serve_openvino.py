import time
import uuid
import json
from threading import Thread
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, TextIteratorStreamer

app = FastAPI()

# Point to your already-downloaded local folder
MODEL_PATH = "./gemma-4-e2b-ov"

print(f"Loading local model from {MODEL_PATH} onto Intel Arc iGPU...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = OVModelForVisualCausalLM.from_pretrained(
    MODEL_PATH,
    device="GPU",
    ov_config={"PERFORMANCE_HINT": "LATENCY"}
)
print("Model ready on Intel GPU!")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 512)
    stream = data.get("stream", False)

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]

    if stream:
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_tokens)
        
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        def event_generator():
            for new_text in streamer:
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:6]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "gemma-4-e2b-it",
                    "choices": [{"delta": {"content": new_text}, "index": 0, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        output = model.generate(**inputs, max_new_tokens=max_tokens)
        response_text = processor.decode(output[0][input_len:], skip_special_tokens=True)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:6]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gemma-4-e2b-it",
            "choices": [{"message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)