"""Use pinned vLLM transport, observing SSE for finish/usage and true TTFT.

The upstream implementation counts role-only chunks as first tokens. The proxy
normalizes UTF-8 SSE boundaries and records content timing without changing the
upstream source. Native timing is retained for comparison.
"""
import asyncio
import codecs
import json
import os
import time
from dataclasses import asdict
from perf_vendor.endpoint_request_func import RequestFuncInput, RequestFuncOutput, async_request_openai_chat_completions


class UsageProtocolError(ValueError):
    pass


def validate_usage(usage):
    if not isinstance(usage,dict):
        raise UsageProtocolError('usage must be an object')
    for key in ('prompt_tokens','completion_tokens','total_tokens'):
        if key in usage:
            value = usage[key]
            if type(value) is not int or value < 0:
                raise UsageProtocolError(f'usage.{key} must be a nonnegative integer')
    if all(key in usage for key in ('prompt_tokens','completion_tokens','total_tokens')):
        if usage['total_tokens'] != usage['prompt_tokens']+usage['completion_tokens']:
            raise UsageProtocolError('usage.total_tokens is inconsistent')
    return usage


class ObservedResponse:
    def __init__(self, response, observation):
        self.response, self.observation = response, observation
        self.status, self.reason = response.status, response.reason
        self.content = self
        observation['http_status'] = response.status

    async def iter_any(self):
        decoder = codecs.getincrementaldecoder('utf-8')()
        buffer = ''
        async for chunk in self.response.content.iter_any():
            buffer += decoder.decode(chunk)
            buffer = buffer.replace('\r\n','\n')
            while '\n\n' in buffer:
                event, buffer = buffer.split('\n\n',1)
                payload = '\n'.join(line[5:].lstrip() for line in event.splitlines() if line.startswith('data:'))
                if not payload: continue
                if payload == '[DONE]':
                    self.observation['done'] = True
                else:
                    data = json.loads(payload)
                    if data.get('usage') is not None:
                        self.observation['usage'] = validate_usage(data['usage'])
                    for choice in data.get('choices',[]):
                        delta = choice.get('delta',{})
                        content = delta.get('content') or delta.get('reasoning_content') or ''
                        if content:
                            self.observation.setdefault('first',time.perf_counter())
                            self.observation['last'] = time.perf_counter()
                            self.observation['progress'](self.observation['last'])
                            self.observation['text'] += content
                        if choice.get('finish_reason'):
                            self.observation['finish_reason'] = choice['finish_reason']
                yield ('data: '+payload+'\n\n').encode()
                if payload == '[DONE]':
                    # End the upstream iterator normally so its response context
                    # releases the HTTP stream without waiting for transport EOF.
                    return
        if buffer.strip():
            raise ValueError('Incomplete SSE event')


class ObservedContext:
    def __init__(self, context, observation): self.context, self.observation = context, observation
    async def __aenter__(self):
        try:
            return ObservedResponse(await self.context.__aenter__(),self.observation)
        except Exception as exc:
            self.observation['error_kind'] = 'timeout' if isinstance(exc, TimeoutError) else 'transport'
            raise
    async def __aexit__(self,*args):
        exc = args[1]
        if exc is not None:
            self.observation['error_kind'] = ('protocol' if isinstance(exc, UsageProtocolError) else
                                              'timeout' if isinstance(exc, TimeoutError) else 'transport')
        return await self.context.__aexit__(*args)


class ObservedSession:
    def __init__(self, session, observation): self.session, self.observation = session, observation
    def post(self, **kwargs): return ObservedContext(self.session.post(**kwargs), self.observation)


class Transport:
    def __init__(self, config, tokenizer, session):
        self.config, self.tokenizer, self.session = config, tokenizer, session

    async def __call__(self, sample):
        c, target = self.config, self.config['target']
        budget = min(c['max_generation_tokens'], c['context_window']-sample['input_tokens'])
        def progress(now): self.last_progress = now
        observation = {'text':'', 'progress':progress}
        native_origin = 'vllm'
        inp = RequestFuncInput(prompt='', api_url=target['base_url'].rstrip('/')+'/chat/completions',
            prompt_len=sample['input_tokens'], output_len=budget, model=target['model'],
            extra_headers={'Authorization':'Bearer '+os.environ[target['api_key_env']]},
            extra_body={**target.get('extra_body',{}), 'messages':sample['messages'],
                        **({'chat_template': self.tokenizer.chat_template} if c.get('chat_template') else {}),
                        'temperature':target.get('temperature',0)})
        started = time.perf_counter()
        task = asyncio.create_task(async_request_openai_chat_completions(inp, ObservedSession(self.session,observation)))
        try:
            while not task.done():
                now = time.perf_counter()
                first_budget = c['timing'].get('first_token_timeout', c['timing']['timeout'])
                idle_budget = c['timing'].get('stream_idle_timeout', c['timing']['no_success'])
                progress_deadline = (observation['last'] + idle_budget) if 'last' in observation else started + first_budget
                deadline = min(started + c['timing']['timeout'], progress_deadline)
                remaining = deadline-now
                if remaining <= 0:
                    observation['error_kind'] = 'timeout'
                    reason = ('request_timeout' if now >= started+c['timing']['timeout'] else
                              'stream_idle_timeout' if 'first' in observation else 'first_token_timeout')
                    observation['timeout_phase'] = reason
                    task.cancel()
                    await asyncio.gather(task,return_exceptions=True)
                    observation['error_kind'] = 'timeout'
                    native = RequestFuncOutput(success=False,error=reason)
                    native_origin = 'adapter_timeout'
                    break
                await asyncio.wait({task}, timeout=min(.05,remaining))
            else:
                native = task.result()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task,return_exceptions=True)
        ended = time.perf_counter()
        first = observation.get('first')
        finish = observation.get('finish_reason')
        protocol_error = None
        try:
            usage = validate_usage(observation.get('usage',{}))
            if first is not None and usage.get('completion_tokens') == 0:
                raise UsageProtocolError('Nonempty output has zero completion_tokens')
            if usage.get('prompt_tokens',0) > c['context_window']:
                raise UsageProtocolError('prompt_tokens exceeds context_window')
            if usage.get('completion_tokens',0) > budget:
                raise UsageProtocolError('completion_tokens exceeds request budget')
        except UsageProtocolError as exc:
            protocol_error = str(exc)
            usage = {}
            observation['error_kind'] = 'protocol'
        # A malformed usage event rejected by the stream parser is also invalid.
        invalid = observation.get('error_kind') == 'protocol'
        count = usage.get('completion_tokens')
        estimated = count is None and not invalid
        if invalid:
            count = 0  # failed request, excluded from throughput/latency statistics
        elif estimated:
            count = len(self.tokenizer.encode(observation['text'],add_special_tokens=False))
        success = bool(native.success and not invalid and observation.get('done') and finish == 'stop' and first is not None)
        ttft = first-started if first is not None else None
        latency = ended-started
        error = protocol_error or native.error
        key = os.environ.get(target['api_key_env'],'')
        if key: error = error.replace(key,'[REDACTED]')
        native_output = asdict(native)
        native_count = 0
        native_source = 'failed'
        if native.success:
            if native.output_tokens:
                native_count, native_source = native.output_tokens, 'usage'
            else:
                # Reuse the suite estimate only when the tokenized text is identical.
                native_source = 'tokenizer'
                native_count = count if estimated and native.generated_text == observation['text'] else len(
                    self.tokenizer.encode(native.generated_text,add_special_tokens=False))
        if key: native_output['error'] = native_output['error'].replace(key,'[REDACTED]')
        return dict(sample_id=sample['id'], start=started, end=ended,
            first_token=first, latency=latency, ttft=ttft,
            tpot=(latency-ttft)/(count-1) if ttft is not None and count>1 else None,
            success=success, finish_reason=finish, truncated=finish=='length',
            input_tokens=usage.get('prompt_tokens',sample['input_tokens']), output_tokens=count,
            input_token_delta=(usage['prompt_tokens']-sample['input_tokens']) if usage.get('prompt_tokens') is not None else None,
            token_source='invalid' if invalid else 'estimated' if estimated else 'usage', http_status=observation.get('http_status'),
            timeout_phase=observation.get('timeout_phase'),
            error_kind=observation.get('error_kind') or (None if success else ('transport' if observation.get('http_status') is None else 'response')),
            error=error or (None if success else 'Incomplete or non-natural completion'),
            native_ttft=native.ttft, native_latency=native.latency,
            native_output=native_output, native_output_origin=native_origin,
            native_input_tokens=sample['input_tokens'],native_report_tokens=native_count,
            native_report_token_source=native_source)
