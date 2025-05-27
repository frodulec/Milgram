import io
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, Response
from fastapi import FastAPI, HTTPException, Body

from loguru import logger
import textwrap
import math
from typing import Tuple, List, Dict, Union
import json
import asyncio
import base64
from utils.chat_utils import load_conversation_dictionary
from utils.drawing_utils import resize_sprite, adjust_cloud

# Add TTS imports
from audio.tts import tts_queue, playback_queue, tts_worker, playback_worker, trigger_next_playback
from models import Roles


def draw_message_on_cloud(
    composite_image: Image.Image, message: str, tail_anchor: Tuple[int, int], flip=False
) -> None:
    """
    Draws the message on the cloud with proper text wrapping.
    """
    cloud = Image.open("static/cloud.png").convert("RGBA")
    cloud = resize_sprite(cloud, 0.15)
    if flip:
        cloud = cloud.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", size=16
    )

    # Adjust the cloud and get wrapped text lines
    cloud, text_lines, line_spacing = adjust_cloud(cloud, message, font)

    # Position the cloud with bottom-left anchor or right-bottom anchor if flipped
    if flip:
        cloud_position = (
            tail_anchor[0] - cloud.size[0],
            tail_anchor[1] - cloud.size[1],
        )
    else:
        cloud_position = (tail_anchor[0], tail_anchor[1] - cloud.size[1])

    composite_image.paste(cloud, cloud_position, cloud)

    # Add text to the cloud
    draw = ImageDraw.Draw(composite_image)

    # Get cloud dimensions
    cloud_width, cloud_height = cloud.size

    # Calculate total text block height
    total_text_height = len(text_lines) * line_spacing

    CLOUD_OFFSET = -11  # because of the tail of the cloud

    start_y = cloud_position[1] + (cloud_height - total_text_height) // 2 + CLOUD_OFFSET

    for i, line in enumerate(text_lines):
        # Calculate bounding box for this specific line
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]

        text_x = cloud_position[0] + (cloud_width - text_width) // 2
        text_y = start_y + (i * line_spacing)
        draw.text((text_x, text_y), line, font=font, fill="black")

    return None


def create_game_image(
    student_message: str | None = None,
    professor_message: str | None = None,
    learner_message: str | None = None,
) -> io.BytesIO:
    """
    Generates the game image by layering sprites on a background.
    Optionally adds a message in a cloud with proper text wrapping.
    """
    background = Image.open("static/background.jpg").convert("RGBA")
    professor_sprite = Image.open("static/professor_w.png").convert("RGBA")
    student_sprite = Image.open("static/student.png").convert("RGBA")
    learner_sprite = Image.open("static/learner.png").convert("RGBA")

    # make the sprites smaller
    professor_sprite = resize_sprite(professor_sprite, 0.58)
    # flip the professor sprite to face the student
    # professor_sprite = professor_sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    student_sprite = resize_sprite(student_sprite, 0.15)

    learner_sprite = resize_sprite(learner_sprite, 1.1)

    # Create a new image canvas to paste everything on
    composite_image = background.copy()

    # 2. Add sprites to the background
    # The third argument is a mask that respects the PNG transparency
    composite_image.paste(professor_sprite, (630, 660), professor_sprite)
    composite_image.paste(student_sprite, (360, 660), student_sprite)
    composite_image.paste(learner_sprite, (673, 275), learner_sprite)

    # 3. Add a message in a cloud if a message is provided
    if student_message:
        draw_message_on_cloud(composite_image, student_message, (450, 670))

    if professor_message:
        draw_message_on_cloud(composite_image, professor_message, (650, 670), True)

    if learner_message:
        draw_message_on_cloud(composite_image, learner_message, (673, 275), True)

    # 4. Save the final image to an in-memory buffer
    img_buffer = io.BytesIO()
    composite_image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    return img_buffer


app = FastAPI()


@app.get("/game-view")
async def get_game_view(
    student_message: str | None = Query(default=None, max_length=1000),
    professor_message: str | None = Query(default=None, max_length=1000),
):
    """
    Endpoint to get the current game view with messages from both characters.
    - Student message: /game-view?student_message=Hello professor!
    - Professor message: /game-view?professor_message=Hello student!
    - Both: /game-view?student_message=Hello!&professor_message=Hi there!
    """
    # Generate the image with specific messages for each character
    image_buffer = create_game_image(student_message, professor_message)
    return StreamingResponse(image_buffer, media_type="image/png")


@app.get("/game-sequence-example")
async def game_sequence_example():
    messages = load_conversation_dictionary("conversation.json")

    async def generate_example_sequence():
        try:
            # Start TTS and playback workers
            tts_task = asyncio.create_task(tts_worker())
            playback_task = asyncio.create_task(playback_worker())

            # Start generating TTS audio
            completion_events = []
            generation_events = []

            for msg in messages:
                speaker_role = None
                if msg["speaker"] == "Participant":
                    speaker_role = Roles.PARTICIPANT
                elif msg["speaker"] == "Professor":
                    speaker_role = Roles.PROFESSOR
                elif msg["speaker"] == "Learner":
                    speaker_role = Roles.LEARNER

                completion_event = asyncio.Event()
                generation_event = asyncio.Event()
                completion_events.append(completion_event)
                generation_events.append(generation_event)

                await tts_queue.put((msg["text"], speaker_role, completion_event, generation_event))

            # Wait for the first audio to be generated
            await generation_events[0].wait()
            
            # Send initial empty image
            image_buffer = create_game_image(None, None)
            image_b64 = base64.b64encode(image_buffer.getvalue()).decode()
            yield f"data: {json.dumps({'image': image_b64, 'step': 0})}\n\n"

            await asyncio.sleep(1.0)  # Initial delay

            for i, msg in enumerate(messages):
                # Reset messages - only show current speaker's message
                current_student_message = None
                current_professor_message = None
                current_learner_message = None

                # Set only the current speaker's message
                if msg["speaker"] == Roles.PARTICIPANT.value:
                    current_student_message = msg["text"]
                elif msg["speaker"] == Roles.PROFESSOR.value:
                    current_professor_message = msg["text"]
                elif msg["speaker"] == Roles.LEARNER.value:
                    current_learner_message = msg["text"]
                
                if current_student_message == "ELECTRIC_SHOCK_IMAGE":
                    pass  # Handle special case

                # Generate and show image with current message
                image_buffer = create_game_image(
                    current_student_message,
                    current_professor_message,
                    current_learner_message,
                )
                image_b64 = base64.b64encode(image_buffer.getvalue()).decode()
                yield f"data: {json.dumps({'image': image_b64, 'step': i + 1})}\n\n"

                # Trigger playback for this message
                await trigger_next_playback()
                
                # Wait for this specific message's audio to finish playing
                await completion_events[i].wait()

            # Wait for any remaining TTS and playback to finish
            await tts_queue.join()
            await playback_queue.join()

            # Cancel worker tasks
            tts_task.cancel()
            playback_task.cancel()

            try:
                await tts_task
                await playback_task
            except asyncio.CancelledError:
                pass

            # Keep connection alive for a bit before closing
            await asyncio.sleep(2.0)
            yield f"data: {json.dumps({'complete': True})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")  # Server-side logging
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_example_sequence(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )

# http://localhost:8000/example
@app.get("/example")
async def make_example():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Game Sequence Example</title>
    </head>
    <body>
        <h1>Game Sequence Example</h1>
        <img id="gameImage" style="max-width: 800px;" />
        <div id="status">Loading...</div>
        
        <script>
            const eventSource = new EventSource('/game-sequence-example');
            const img = document.getElementById('gameImage');
            const status = document.getElementById('status');
            
            eventSource.onopen = function(event) {
                status.textContent = 'Connected, waiting for images...';
                console.log('EventSource opened');
            };
            
            eventSource.onmessage = function(event) {
                console.log('Received:', event.data);
                if (event.data === 'image_update') {
                    status.textContent = 'Starting sequence...';
                } else {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.image) {
                            img.src = 'data:image/png;base64,' + data.image;
                            status.textContent = `Step ${data.step}`;
                        }
                        if (data.complete) {
                            status.textContent = 'Sequence complete!';
                            eventSource.close();
                        }
                    } catch (e) {
                        console.error('Parse error:', e);
                    }
                }
            };
            
            eventSource.onerror = function(event) {
                console.error('EventSource error:', event);
                status.textContent = 'Error occurred - check console';
            };
        </script>
    </body>
    </html>
    """
    return Response(html_content, media_type="text/html")
