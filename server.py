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
from audio.tts import (
    generate_tts,
    tts_worker,
    playback_worker,
    trigger_next_playback,
)
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
    display_shock: bool = False,
) -> io.BytesIO:
    """
    Generates the game image by layering sprites on a background.
    Optionally adds a message in a cloud with proper text wrapping.
    """
    background = Image.open("static/background.jpg").convert("RGBA")
    professor_sprite = Image.open("static/professor_w.png").convert("RGBA")
    student_sprite = Image.open("static/student.png").convert("RGBA")
    learner_sprite = Image.open("static/learner.png").convert("RGBA")
    shock_sprite = Image.open("static/electricity.png").convert("RGBA")

    # make the sprites smaller
    professor_sprite = resize_sprite(professor_sprite, 0.58)
    # flip the professor sprite to face the student
    # professor_sprite = professor_sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    student_sprite = resize_sprite(student_sprite, 0.15)
    learner_sprite = resize_sprite(learner_sprite, 1.1)
    shock_sprite = resize_sprite(shock_sprite, 0.05)

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

    if display_shock:
        composite_image.paste(shock_sprite, (673, 300), shock_sprite)

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
            # Generate all audio files first
            audio_data_list = []
            audio_durations = []  # Store audio durations
            
            for msg in messages:
                speaker_role = None
                if msg["speaker"] == "Participant":
                    speaker_role = Roles.PARTICIPANT
                elif msg["speaker"] == "Professor":
                    speaker_role = Roles.PROFESSOR
                elif msg["speaker"] == "Learner":
                    speaker_role = Roles.LEARNER

                if speaker_role is None:
                    audio_data_list.append(None)
                    audio_durations.append(0)
                    continue

                # Generate TTS audio
                audio_data = await generate_tts(msg["text"], speaker_role)
                
                # Get bytes from BytesIO object
                if hasattr(audio_data, 'getvalue'):
                    audio_bytes = audio_data.getvalue()
                else:
                    audio_bytes = audio_data
                
                # Calculate audio duration
                try:
                    import io
                    import tempfile
                    import os
                    from pydub import AudioSegment
                    
                    # Create a temporary file to save the audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                        temp_file.write(audio_bytes)
                        temp_file.flush()
                        
                        # Load audio and get duration
                        audio_segment = AudioSegment.from_file(temp_file.name)
                        duration = len(audio_segment) / 1000.0  # Convert milliseconds to seconds
                        
                        # Clean up temp file
                        os.unlink(temp_file.name)
                    
                    audio_durations.append(duration)
                except Exception as e:
                    logger.warning(f"Could not calculate audio duration: {e}")
                    # Fallback to estimated duration based on text length
                    # Average speaking rate is about 150-160 words per minute, or ~2.5 words per second
                    word_count = len(msg["text"].split())
                    estimated_duration = max(1.0, word_count / 2.5)
                    audio_durations.append(estimated_duration)
                
                # Convert audio bytes to base64
                audio_b64 = base64.b64encode(audio_bytes).decode()
                audio_data_list.append(audio_b64)

            # Send initial empty image
            image_buffer = create_game_image(None, None)
            image_b64 = base64.b64encode(image_buffer.getvalue()).decode()
            yield f"data: {json.dumps({'image': image_b64, 'step': 0})}\n\n"

            await asyncio.sleep(1.0)  # Initial delay

            for i, msg in enumerate(messages):
                current_student_message = None
                current_professor_message = None
                current_learner_message = None

                if msg["speaker"] == "SHOCKING_DEVICE":
                    # add shock image, remove, add shock image and remove
                    for _ in range(2):
                        image_buffer = create_game_image(None, None, None, True)
                        image_b64 = base64.b64encode(image_buffer.getvalue()).decode()
                        yield f"data: {json.dumps({'image': image_b64, 'step': i + 1})}\n\n"
                        await asyncio.sleep(0.5)
                        image_buffer = create_game_image(None, None, None, False)
                        image_b64 = base64.b64encode(image_buffer.getvalue()).decode()
                        yield f"data: {json.dumps({'image': image_b64, 'step': i + 1})}\n\n"
                        await asyncio.sleep(0.5)
                    continue

                # Set only the current speaker's message
                if msg["speaker"] == Roles.PARTICIPANT.value:
                    current_student_message = msg["text"]
                elif msg["speaker"] == Roles.PROFESSOR.value:
                    current_professor_message = msg["text"]
                elif msg["speaker"] == Roles.LEARNER.value:
                    current_learner_message = msg["text"]

                # Generate and show image with current message
                image_buffer = create_game_image(
                    current_student_message,
                    current_professor_message,
                    current_learner_message,
                )
                image_b64 = base64.b64encode(image_buffer.getvalue()).decode()
                
                # Send image first
                yield f"data: {json.dumps({'image': image_b64, 'step': i + 1})}\n\n"
                
                # Small delay to ensure image is displayed before audio starts
                await asyncio.sleep(0.1)
                
                # Then send audio if available
                if audio_data_list[i] is not None:
                    audio_response = {'audio': audio_data_list[i], 'step': i + 1}
                    yield f"data: {json.dumps(audio_response)}\n\n"
                    
                    # Wait for actual audio duration plus small buffer
                    audio_duration = audio_durations[i]
                    await asyncio.sleep(audio_duration + 0.5)  # Add 0.5s buffer
                else:
                    # If no audio, just wait a bit before next message
                    await asyncio.sleep(1.0)

            # Keep connection alive for a bit before closing
            await asyncio.sleep(2.0)
            yield f"data: {json.dumps({'complete': True})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
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
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            #startButton { 
                padding: 15px 30px; 
                font-size: 18px; 
                background-color: #4CAF50; 
                color: white; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer; 
                margin-bottom: 20px;
            }
            #startButton:hover { background-color: #45a049; }
            #startButton:disabled { 
                background-color: #cccccc; 
                cursor: not-allowed; 
            }
            #controls { margin-bottom: 20px; }
            .control-button {
                padding: 10px 20px;
                margin: 5px;
                font-size: 14px;
                border: none;
                border-radius: 3px;
                cursor: pointer;
            }
            #muteButton { background-color: #ff9800; color: white; }
            #volumeControl { margin-left: 10px; }
        </style>
    </head>
    <body>
        <h1>Game Sequence Example</h1>
        
        <div id="controls">
            <button id="startButton">Start Game Sequence</button>
            <button id="muteButton" class="control-button" style="display: none;">Mute Audio</button>
            <input type="range" id="volumeControl" min="0" max="1" step="0.1" value="1" style="display: none;">
            <span id="volumeLabel" style="display: none;">Volume</span>
        </div>
        
        <img id="gameImage" style="max-width: 800px;" />
        <div id="status">Click "Start Game Sequence" to begin</div>
        <audio id="audioPlayer" preload="auto"></audio>
        
        <script>
            let eventSource = null;
            let audioQueue = [];
            let isPlaying = false;
            let isMuted = false;
            
            const startButton = document.getElementById('startButton');
            const muteButton = document.getElementById('muteButton');
            const volumeControl = document.getElementById('volumeControl');
            const volumeLabel = document.getElementById('volumeLabel');
            const img = document.getElementById('gameImage');
            const status = document.getElementById('status');
            const audioPlayer = document.getElementById('audioPlayer');
            
            function base64ToBlob(base64, mimeType) {
                const byteCharacters = atob(base64);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                return new Blob([byteArray], { type: mimeType });
            }
            
            async function playAudio(audioData) {
                if (isMuted) return;
                
                try {
                    // Wait for any currently playing audio to finish
                    if (isPlaying) {
                        audioQueue.push(audioData);
                        return;
                    }
                    
                    isPlaying = true;
                    
                    // Create blob from base64 audio data
                    const audioBlob = base64ToBlob(audioData, 'audio/wav');
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    audioPlayer.src = audioUrl;
                    audioPlayer.volume = volumeControl.value;
                    
                    await audioPlayer.play();
                    
                    // Clean up URL after playing
                    audioPlayer.onended = () => {
                        URL.revokeObjectURL(audioUrl);
                        isPlaying = false;
                        
                        // Play next audio in queue if any
                        if (audioQueue.length > 0) {
                            const nextAudio = audioQueue.shift();
                            playAudio(nextAudio);
                        }
                    };
                } catch (error) {
                    console.error('Audio playback error:', error);
                    isPlaying = false;
                }
            }
            
            function startSequence() {
                startButton.disabled = true;
                startButton.textContent = 'Running...';
                
                // Show audio controls
                muteButton.style.display = 'inline-block';
                volumeControl.style.display = 'inline-block';
                volumeLabel.style.display = 'inline-block';
                
                eventSource = new EventSource('/game-sequence-example');
                
                eventSource.onopen = function(event) {
                    status.textContent = 'Connected, waiting for images...';
                    console.log('EventSource opened');
                };
                
                eventSource.onmessage = function(event) {
                    console.log('Received:', event.data);
                    try {
                        const data = JSON.parse(event.data);
                        
                        if (data.image) {
                            img.src = 'data:image/png;base64,' + data.image;
                            status.textContent = `Step ${data.step}`;
                        }
                        
                        if (data.audio) {
                            // Add audio to queue for playback
                            playAudio(data.audio);
                        }
                        
                        if (data.complete) {
                            status.textContent = 'Sequence complete!';
                            startButton.textContent = 'Start Game Sequence';
                            startButton.disabled = false;
                            eventSource.close();
                        }
                        
                        if (data.error) {
                            status.textContent = `Error: ${data.error}`;
                            startButton.textContent = 'Start Game Sequence';
                            startButton.disabled = false;
                            eventSource.close();
                        }
                    } catch (e) {
                        console.error('Parse error:', e);
                    }
                };
                
                eventSource.onerror = function(event) {
                    console.error('EventSource error:', event);
                    status.textContent = 'Error occurred - check console';
                    startButton.textContent = 'Start Game Sequence';
                    startButton.disabled = false;
                };
            }
            
            // Event listeners
            startButton.addEventListener('click', startSequence);
            
            muteButton.addEventListener('click', function() {
                isMuted = !isMuted;
                muteButton.textContent = isMuted ? 'Unmute Audio' : 'Mute Audio';
                muteButton.style.backgroundColor = isMuted ? '#f44336' : '#ff9800';
                
                if (isMuted && !audioPlayer.paused) {
                    audioPlayer.pause();
                }
            });
            
            volumeControl.addEventListener('input', function() {
                audioPlayer.volume = this.value;
            });
        </script>
    </body>
    </html>
    """
    return Response(html_content, media_type="text/html")