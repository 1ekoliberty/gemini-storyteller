"""Private, single-user Gemini narration bridge for ChatGPT and Render."""
import asyncio
import base64
import contextlib
import io
import os
import re
import secrets
import time
import wave
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route

API_KEY = os.environ.get('GEMINI_API_KEY', '')
SECRET = os.environ.get('ACCESS_SECRET', '')
BASE_URL = os.environ.get('RENDER_EXTERNAL_URL', os.environ.get('BASE_URL', '')).rstrip('/')
MODEL = os.environ.get('GEMINI_MODEL', 'gemini-3.8-flash-tts')
VOICE = os.environ.get('GEMINI_VOICE', 'Sulafat')
STYLE = os.environ.get('NARRATOR_STYLE', 'Read in Russian as a warm, gentle storyteller reading to a child. Natural expressive intonation, relaxed but lively pace, meaningful pauses. Preserve every word of the text.')
STORE = Path('/tmp/storyteller-audio')
STORE.mkdir(exist_ok=True)
jobs = {}
tasks = set()

def chunks(text, limit=3000):
    result = []
    while len(text) > limit:
        cut = max(text.rfind('\n', 0, limit), text.rfind('. ', 0, limit), text.rfind('! ', 0, limit), text.rfind('? ', 0, limit))
        cut = cut + 1 if cut > limit // 3 else text.rfind(' ', 0, limit)
        if cut <= 0:
            cut = limit
        result.append(text[:cut])
        text = text[cut:]
    if text:
        result.append(text)
    return result

def extract_audio(data):
    for step in reversed(data.get('steps', [])):
        for content in step.get('content', []):
            if content.get('type') == 'audio' and content.get('data'):
                raw = base64.b64decode(content['data'], validate=True)
                if not raw.startswith(b'RIFF'):
                    raise ValueError('Expected WAV audio')
                return raw
    raise ValueError('No audio in response')

def merge_wav(parts):
    output = io.BytesIO()
    expected = None
    with wave.open(output, 'wb') as writer:
        for part in parts:
            with wave.open(io.BytesIO(part), 'rb') as reader:
                params = (reader.getnchannels(), reader.getsampwidth(), reader.getframerate())
                if expected is None:
                    expected = params
                    writer.setnchannels(params[0])
                    writer.setsampwidth(params[1])
                    writer.setframerate(params[2])
                if expected != params:
                    raise ValueError('Incompatible audio formats')
                writer.writeframes(reader.readframes(reader.getnframes()))
    return output.getvalue()

def cleanup():
    now = time.time()
    for job_id, job in list(jobs.items()):
        if job['status'] != 'processing' and now - job['created'] > 3600:
            (STORE / (job_id + '.wav')).unlink(missing_ok=True)
            del jobs[job_id]

async def generate(job_id, text):
    job = jobs[job_id]
    try:
        parts = []
        async with asyncio.timeout(1200):
            async with httpx.AsyncClient(timeout=240) as client:
                for section in chunks(text):
                    response = await client.post(
                        'https://generativelanguage.googleapis.com/v1beta/interactions',
                        headers={'x-goog-api-key': API_KEY},
                        json={
                            'model': MODEL,
                            'input': [{'type': 'user_input', 'content': [{'type': 'text', 'text': section, 'annotations': [{'type': 'speech_metadata', 'style': STYLE}]}]}],
                            'response_format': {'type': 'audio'},
                            'generation_config': {'speech_config': [{'voice': VOICE}]},
                        })
                    if response.status_code != 200:
                        job.update(status='error', message={400: 'Google отклонил настройки модели или голоса.', 401: 'Проверьте ключ Gemini в Render.', 403: 'Google запретил доступ: проверьте ключ и доступность модели.', 404: 'Эта модель недоступна для вашего проекта.', 429: 'Лимит Google исчерпан. Повторите позже.'}.get(response.status_code, 'Ошибка Google. Повторите позже.'))
                        return
                    parts.append(extract_audio(response.json()))
                    if sum(map(len, parts)) > 100_000_000:
                        raise ValueError('Audio too large')
                    job['completed_parts'] += 1
        (STORE / (job_id + '.wav')).write_bytes(merge_wav(parts))
        job['status'] = 'ready'
    except asyncio.CancelledError:
        job.update(status='error', message='Сервис перезапущен. Создайте озвучку заново.')
        raise
    except Exception:
        job.update(status='error', message='Не удалось получить полноценную запись. Попробуйте короткий фрагмент или проверьте настройки модели и голоса.')

mcp = FastMCP('Рассказчик Gemini', stateless_http=True, json_response=True,
    instructions='Generate Russian narration only when requested. Use start_narration, then check_narration. Never claim an audio file exists until status is ready. Show the returned download_url as a clickable link. Jobs and files are temporary; tell the user to download the file. Do not request or reveal API keys or the private MCP URL.',
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=['localhost:*', '127.0.0.1:*', 'testserver', BASE_URL.removeprefix('https://').removeprefix('http://')], allowed_origins=[]))

@mcp.tool(annotations={'readOnlyHint': False, 'destructiveHint': False, 'idempotentHint': False, 'openWorldHint': True})
async def start_narration(text: str) -> dict:
    """Озвучить точный текст одним рассказчиком. Передаёт текст в Google Gemini. Вернёт job_id; затем проверяйте check_narration. До 20000 символов."""
    cleanup()
    if not API_KEY or not BASE_URL:
        return {'error': 'Добавьте GEMINI_API_KEY и проверьте адрес Render.'}
    if not text.strip() or len(text) > 20000:
        return {'error': 'Нужен текст от 1 до 20000 символов.'}
    if any(j['status'] == 'processing' for j in jobs.values()):
        return {'error': 'Одна запись уже обрабатывается. Дождитесь её завершения.'}
    if len(jobs) >= 6:
        return {'error': 'Достигнут лимит временных записей. Попробуйте через час.'}
    job_id = secrets.token_urlsafe(24)
    jobs[job_id] = {'status': 'processing', 'created': time.time(), 'completed_parts': 0, 'total_parts': len(chunks(text))}
    task = asyncio.create_task(generate(job_id, text))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return {'job_id': job_id, 'status': 'processing', 'message': 'Озвучка запущена. Проверьте готовность через 20–30 секунд.'}

@mcp.tool(annotations={'readOnlyHint': True, 'destructiveHint': False, 'idempotentHint': True, 'openWorldHint': False})
def check_narration(job_id: str) -> dict:
    """Проверить готовность озвучки по job_id. При ready вернуть пользователю ссылку на WAV."""
    cleanup()
    job = jobs.get(job_id)
    if not job:
        return {'error': 'Запись не найдена или сервис перезапущен. Создайте озвучку заново.'}
    result = {k: v for k, v in job.items() if k != 'created'}
    if job['status'] == 'ready':
        result['download_url'] = f'{BASE_URL}/audio/{job_id}.wav'
        result['message'] = 'Скачайте WAV сейчас: ссылка действует до часа, но исчезнет раньше при остановке или перезапуске Render.'
    return result

async def health(request):
    return JSONResponse({'status': 'ok'})

async def audio(request):
    name = request.path_params['name']
    if not re.fullmatch(r'[A-Za-z0-9_-]{32}\.wav', name):
        return JSONResponse({'error': 'Not found'}, status_code=404)
    cleanup()
    path = STORE / name
    if not path.is_file():
        return JSONResponse({'error': 'Expired or not found'}, status_code=404)
    return FileResponse(path, media_type='audio/wav', filename='skazka.wav', headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'})

@contextlib.asynccontextmanager
async def lifespan(app):
    if len(SECRET) < 32 or not re.fullmatch(r'[A-Za-z0-9_-]+', SECRET):
        raise RuntimeError('Set ACCESS_SECRET to at least 32 random URL-safe characters in Render.')
    async with mcp.session_manager.run():
        yield
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

app = Starlette(routes=[Route('/', health), Route('/health', health), Route('/audio/{name}', audio), Mount('/' + (SECRET or 'unconfigured'), app=mcp.streamable_http_app())], lifespan=lifespan)
