"""Local OpenAI-compatible SSE fixture; never use for performance comparisons."""
import asyncio
import json
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained(os.environ['MOCK_TOKENIZER'], local_files_only=True)
slot = asyncio.Semaphore(1)
stats = dict(requests=0, active=0, peak_active=0, waiting=0, peak_waiting=0)

@app.get('/health')
async def health():
    return dict(status='ok', **stats)

@app.post('/v1/chat/completions')
async def completion(request: Request):
    body = await request.json()
    mode = body.get('mock_mode', 'normal')
    stats['requests'] += 1
    if mode in ('429', '503'):
        return JSONResponse({'error': {'message': 'injected failure'}}, status_code=int(mode))
    prompt = len(tokenizer.apply_chat_template(body['messages'], tokenize=True, add_generation_prompt=True))
    def event(value):
        return ('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode()
    async def stream():
        stats['waiting'] += 1
        stats['peak_waiting'] = max(stats['peak_waiting'], stats['waiting'])
        acquired = False
        try:
            await slot.acquire()
            acquired = True
            stats['waiting'] -= 1
            stats['active'] += 1
            stats['peak_active'] = max(stats['peak_active'], stats['active'])
            yield event({'choices': [{'delta': {'role': 'assistant'}, 'finish_reason': None}]})
            if mode == 'first_timeout':
                await asyncio.sleep(2)
            for i in range(16):
                await asyncio.sleep(.01)
                yield event({'choices': [{'delta': {'content': 'x'}, 'finish_reason': None}]})
                if mode == 'idle_timeout':
                    await asyncio.sleep(2)
            if mode == 'incomplete':
                return
            if mode != 'done_without_finish':
                yield event({'choices': [{'delta': {}, 'finish_reason': 'length' if mode == 'length' else 'stop'}]})
            if mode != 'no_usage':
                usage = dict(prompt_tokens=prompt + (1 if mode == 'mismatch' else 0),
                             completion_tokens=-1 if mode == 'invalid_usage' else 16)
                yield event({'choices': [], 'usage': usage})
            yield b'data: [DONE]\n\n'
            if mode == 'done_open':
                await asyncio.sleep(2)
        finally:
            if acquired:
                stats['active'] -= 1
                slot.release()
            else:
                stats['waiting'] -= 1
    return StreamingResponse(stream(), media_type='text/event-stream')
