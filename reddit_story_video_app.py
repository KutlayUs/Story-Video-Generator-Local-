import streamlit as st
import tempfile
import subprocess
import os
import shutil
import sys
import requests
import time
import logging
import re
from pathlib import Path
import json

# Import PIL for direct image rendering
from PIL import Image, ImageDraw, ImageFont

# Import numpy for converting PIL images to MoviePy compatible arrays
import numpy as np

# MoviePy imports are here but will be checked and imported again for safety later
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ColorClip
from moviepy.video.VideoClip import ImageClip # We'll use ImageClip for rendered text


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to convert hex color to RGB tuple
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Dependency check functions
def is_command_available(cmd):
    return shutil.which(cmd) is not None

def is_python_package_installed(pkg):
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Text preprocessing for TTS
def preprocess_text_for_tts(text):
    """
    Preprocess text to make it more suitable for TTS.
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Ensure minimum length (adjust if needed)
    if len(text) < 50: 
        # Pad with additional context if too short
        text = f"Here's an interesting story for you. {text} That's the end of this story."
    
    # Add periods for better speech synthesis
    if not text.endswith('.') and not text.endswith('!') and not text.endswith('?'):
        text += '.'
    
    # Replace problematic characters
    text = text.replace('\n', '. ')
    text = text.replace('\t', ' ')
    # Allow a broader range of characters often found in Reddit posts, but still clean
    text = re.sub(r'[^\w\s.,!?;:\-\'\"#@$%&*\(\)\[\]\{\}\\/]', '', text) 
    
    # Ensure no double periods from replacements or existing text
    text = re.sub(r'(\.\s*){2,}', '. ', text)
    
    return text

# Font verification function (now simplified as Pillow will handle the direct rendering)
_moviepy_imported_for_font_check = False
def verify_font_file(font_path):
    """
    Verify that the font file can be loaded by Pillow.
    """
    try:
        ImageFont.truetype(font_path, 1) # Try to load a tiny size
        return True
    except Exception as e:
        logger.error(f"Font verification failed for {font_path}: {e}")
        return False

# Dependency checks
missing = []
installable = []

if not is_python_package_installed("TTS"):
    missing.append("TTS (pip install TTS)")
    installable.append(("TTS", "pip"))
if not is_command_available("whisper"):
    missing.append("whisper (pip install openai-whisper)")
    installable.append(("openai-whisper", "pip"))
if not is_command_available("ffmpeg"):
    missing.append("ffmpeg (https://ffmpeg.org/download.html)")
if not is_command_available("ollama"):
    missing.append("ollama (https://ollama.com/download)")
if not is_python_package_installed("moviepy"):
    missing.append("moviepy (pip install moviepy)")
    installable.append(("moviepy", "pip"))
if not is_python_package_installed("PIL"): # Corrected for Pillow module name
    missing.append("Pillow (pip install Pillow)")
    installable.append(("Pillow", "pip")) 
if not is_python_package_installed("numpy"):
    missing.append("numpy (pip install numpy)")
    installable.append(("numpy", "pip"))


if missing:
    st.error("Missing dependencies:\n" + "\n".join(f"- {m}" for m in missing))
    for pkg, method in installable:
        if st.button(f"Install {pkg}"):
            with st.spinner(f"Installing {pkg}..."):
                try:
                    pip_install(pkg)
                    st.success(f"{pkg} installed! Please restart the app.")
                except Exception as e:
                    st.error(f"Failed to install {pkg}: {e}")
    if not is_command_available("ollama"):
        st.markdown(
            "[Download Ollama and follow install instructions](https://ollama.com/download)"
        )
    if not is_command_available("ffmpeg"):
        st.markdown(
            "[Download FFmpeg and add to PATH](https://ffmpeg.org/download.html)"
        )
    st.stop()

# Import TTS after dependency check
try:
    from TTS.api import TTS
except ImportError as e:
    st.error(f"Failed to import TTS: {e}")
    st.stop()
# Import MoviePy after dependency check
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ColorClip
    from moviepy.video.VideoClip import ImageClip # We'll use ImageClip for rendered text
    _moviepy_imported_for_font_check = True # Confirm MoviePy components are available now
except ImportError as e:
    st.error(f"Failed to import MoviePy components: {e}")
    st.stop()


st.title("Reddit-like Story Video Generator (Ollama LLM)")

# Initialize session state
if 'generated_story' not in st.session_state:
    st.session_state.generated_story = ""

# --- Step 1: User Inputs ---
st.header("1. Story Generation")
topic = st.text_input("Enter a topic for the Reddit-like story:")

# --- Step 1b: Ollama LLM Selection ---
def get_ollama_models():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e: # Catch all exceptions from requests
        logger.error(f"Failed to get Ollama models: {e}")
        return []

ollama_models = get_ollama_models()
if ollama_models:
    llm_model = st.selectbox("Select Ollama LLM model:", ollama_models)
else:
    llm_model = st.text_input("Ollama model name (e.g., llama3, phi3, mistral):", value="llama3")
    if not ollama_models:
        st.warning("Could not connect to Ollama. Make sure it's running on localhost:11434")

# --- Step 2: Text-to-Speech (Coqui TTS) ---
st.header("2. Text-to-Speech (Coqui TTS)")
tts_models = [
    "tts_models/en/ljspeech/tacotron2-DDC",
    "tts_models/en/ljspeech/glow-tts",
    "tts_models/en/ljspeech/speedy-speech",
    "tts_models/en/ljspeech/neural_hmm",
    "tts_models/en/ljspeech/tacotron2-DCA",
    "tts_models/en/ljspeech/overflow",
]
tts_model = st.selectbox("Select TTS model:", tts_models)
# Add speech speed control
speech_speed = st.slider(
    "Speech Speed:", 
    min_value=0.5, 
    max_value=2.0, 
    value=1.0, 
    step=0.05,
    help="Control the speed of speech (1.0 is normal, <1.0 is slower, >1.0 is faster)."
)

# --- Step 3: Whisper Model Selection ---
st.header("3. Speech-to-Text (Whisper)")
whisper_models = ["tiny", "base", "small", "medium", "large"]
whisper_model = st.selectbox("Select Whisper model:", whisper_models, index=2)

# --- Step 4: Video Upload ---
st.header("4. Background Video")
video_file = st.file_uploader("Upload a background video (mp4):", type=["mp4"])

# --- Step 5: Output File Names & Location ---
st.header("5. Output Settings")
output_video_name = st.text_input("Output video filename:", value="final_video.mp4")

# Let user pick output location
output_dir = st.text_input("Output folder (absolute path, leave blank for default temp folder):", value="")


# --- Step 6: Subtitle Appearance ---
st.header("6. Subtitle Appearance")
# Font Upload
uploaded_font = st.file_uploader("Upload custom font (.ttf, .otf):", type=["ttf", "otf"])

# Font Parameters
col1, col2 = st.columns(2)
with col1:
    font_size = st.number_input("Font Size:", min_value=10, max_value=100, value=40)
    text_color = st.color_picker("Text Color:", value="#FFFFFF") # White (hex)
    stroke_color = st.color_picker("Outline Color:", value="#000000") # Black (hex)
    stroke_width = st.number_input("Outline Width:", min_value=0, max_value=5, value=2)
with col2:
    bg_color_rgb = st.color_picker("Background Color:", value="#000000") # Black (hex)
    bg_opacity = st.slider("Background Opacity:", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    padding = st.number_input("Padding (around text):", min_value=0, max_value=20, value=10)


# Function to render text onto a Pillow image
def render_text_to_pillow_image(
    text, 
    font_path, 
    font_size, 
    text_color_hex, 
    stroke_color_hex, 
    stroke_width_pixels, 
    bg_color_hex, 
    bg_opacity_float, 
    padding_pixels,
    max_width_pixels # Maximum width for text wrapping
):
    try:
        # Load font. If font_path is None or invalid, try a common system font.
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                # Fallback to common system fonts if custom font is not provided or invalid
                # This might vary across OS. Arial is common on Windows, DejaVuSans on Linux.
                if sys.platform == "win32":
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
                    logger.warning(f"Using fallback font: C:/Windows/Fonts/arial.ttf for text '{text}'")
                elif sys.platform == "darwin": # macOS
                    font = ImageFont.truetype("/Library/Fonts/Arial.ttf", font_size)
                    logger.warning(f"Using fallback font: /Library/Fonts/Arial.ttf for text '{text}'")
                else: # Linux and others
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                    logger.warning(f"Using fallback font: /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf for text '{text}'")
        except Exception as e:
            logger.error(f"Could not load specified or fallback font, using default Pillow font: {e}")
            font = ImageFont.load_default() # Last resort fallback

        # Create a dummy image and draw object to calculate text size
        dummy_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw_dummy = ImageDraw.Draw(dummy_img)

        # Calculate text bounding box including stroke
        # textbbox gives (left, top, right, bottom)
        bbox = draw_dummy.textbbox((0, 0), text, font=font, stroke_width=stroke_width_pixels)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate image size with padding and ensure min size
        img_width = text_width + 2 * padding_pixels
        img_height = text_height + 2 * padding_pixels
        
        # Ensure image width doesn't exceed max_width_pixels if it's too large initially
        if img_width > max_width_pixels:
            img_width = max_width_pixels # Cap it if the word itself is too long

        # Create a transparent image
        img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw background rectangle
        if bg_opacity_float > 0:
            r_bg, g_bg, b_bg = hex_to_rgb(bg_color_hex)
            alpha_bg = int(255 * bg_opacity_float)
            draw.rectangle(
                [(0, 0), (img_width, img_height)], 
                fill=(r_bg, g_bg, b_bg, alpha_bg)
            )

        # Draw text
        text_color_rgb = hex_to_rgb(text_color_hex)
        stroke_color_rgb = hex_to_rgb(stroke_color_hex)

        # Position text in the center of the image
        text_x = padding_pixels - bbox[0] # Adjust for potential negative bbox[0]
        text_y = padding_pixels - bbox[1] # Adjust for potential negative bbox[1]

        draw.text(
            (text_x, text_y), 
            text, 
            font=font, 
            fill=text_color_rgb, 
            stroke_width=stroke_width_pixels, 
            stroke_fill=stroke_color_rgb
        )
        
        return img

    except Exception as e:
        logger.error(f"Error rendering text '{text}' with Pillow: {e}")
        st.warning(f"Could not render text '{text}' with specified font/settings. Using basic rendering.")
        # Fallback to a basic solid color image if all else fails
        fallback_img = Image.new('RGBA', (100, 50), (255,0,0,255)) # Red solid image for error visibility
        draw_fallback = ImageDraw.Draw(fallback_img)
        draw_fallback.text((10,10), "?", fill=(0,0,0))
        return fallback_img


# --- Step 7: Generate Story First ---
st.header("7. Generate Story")
if st.button("Generate Story"):
    if not topic:
        st.error("Please enter a topic for the story.")
    else:
        with st.spinner("Generating story with Ollama..."):
            prompt = f"""Write a compelling Reddit-style story about {topic}. 
            Make it engaging, detailed, and at least 200 words long. 
            Include dialogue and descriptive details that would work well for narration.
            Format it as a continuous narrative without special Reddit formatting."""
            
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": llm_model, 
                        "prompt": prompt, 
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "top_k": 40
                        }
                    },
                    timeout=120
                )
                response.raise_for_status() # Corrected line
                story = response.json().get("response", "").strip()
                
                if story:
                    st.session_state.generated_story = story
                    st.success("Story generated successfully!")
                else:
                    st.error("Generated story is empty. Please try again.")
                    
            except Exception as e:
                st.error(f"Failed to generate story with Ollama: {e}")

# Display generated story
if st.session_state.generated_story:
    st.subheader("Generated Story")
    st.text_area("Story Content", st.session_state.generated_story, height=300, key="story_display")
    
    # Option to edit the story
    if st.checkbox("Edit story before generating video"):
        edited_story = st.text_area("Edit your story:", st.session_state.generated_story, height=200, key="story_edit")
        if st.button("Update Story"):
            st.session_state.generated_story = edited_story
            st.success("Story updated!")

# --- Step 8: Generate Video ---
st.header("8. Generate Video")
if st.button("Generate Video"):
    if not all([st.session_state.generated_story, llm_model, tts_model, whisper_model, video_file]):
        st.error("Please ensure a story is generated, all models are selected, and a background video is uploaded.")
        st.stop()
    else:
        story = st.session_state.generated_story
        processed_story = preprocess_text_for_tts(story)
        
        if len(processed_story.strip()) < 50:
            st.error("Story is too short for TTS processing even after padding. Please generate or edit to a longer story.")
            st.stop()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                story_txt = os.path.join(tmpdir, "story.txt")
                with open(story_txt, "w", encoding="utf-8") as f:
                    f.write(processed_story)

                # Improved font path handling
                font_path_for_moviepy = None  # Start with None for MoviePy's default

                if uploaded_font:
                    font_filename = uploaded_font.name
                    # Check if the uploaded file has a valid font extension
                    if not (font_filename.lower().endswith('.ttf') or font_filename.lower().endswith('.otf')):
                        st.warning(f"Uploaded file '{font_filename}' is not a valid font type (.ttf or .otf). Attempting to use a system font.")
                    else:
                        font_save_path = os.path.join(tmpdir, font_filename)
                        try:
                            # Reset file pointer to beginning
                            uploaded_font.seek(0)
                            with open(font_save_path, "wb") as f:
                                f.write(uploaded_font.read())
                            
                            # Verify the font file was saved and has content
                            if os.path.exists(font_save_path) and os.path.getsize(font_save_path) > 0:
                                # Test the font before using it with Pillow
                                if verify_font_file(font_save_path):
                                    font_path_for_moviepy = font_save_path # Renamed variable for clarity
                                    st.success(f"Custom font '{font_filename}' verified and will be used for subtitles.")
                                else:
                                    st.error(f"Custom font '{font_filename}' failed verification. This might be due to a corrupt or incompatible font file. Using a system font fallback.")
                                    font_path_for_moviepy = None # Explicitly set to None
                            else:
                                st.error("Font file was not saved properly or is empty. Using a system font fallback.")
                                
                        except Exception as e:
                            st.error(f"Failed to save or process custom font file: {e}. Using a system font fallback.")
                else:
                    st.info("No custom font uploaded. A common system font will be used for subtitles.")

                # Step 2: Text to Speech (Coqui TTS)
                story_wav = os.path.join(tmpdir, "story.wav")
                st.info("Converting story to speech...")
                progress_bar = st.progress(0, text="Initializing TTS model...")
                
                try:
                    current_tts_model = tts_model if tts_model else "tts_models/en/ljspeech/tacotron2-DDC"
                    tts = TTS(current_tts_model, progress_bar=False)
                    progress_bar.progress(30, text="Generating audio...")
                    # Pass the speech_speed parameter here!
                    tts.tts_to_file(text=processed_story, file_path=story_wav, speed=speech_speed) 
                    progress_bar.progress(50, text="Audio generated successfully!")
                    
                    if not os.path.exists(story_wav) or os.path.getsize(story_wav) == 0:
                        raise Exception("Audio file was not generated or is empty. This can happen with very short stories or specific model issues.")
                    st.success("Audio generated successfully!")
                    
                except Exception as e:
                    st.error(f"TTS failed: {e}")
                    st.info("Try using a different TTS model or ensure your story is longer and contains more varied text.")
                    st.stop()

                # Step 3: Audio to Subtitles (Whisper) - Now outputting JSON for word-level timestamps
                st.info("Generating word-level subtitles...")
                progress_bar.progress(60, text="Generating word-level subtitles with Whisper...")
                
                word_timestamps_json = os.path.join(tmpdir, "story.json") # Output JSON file
                
                try:
                    current_whisper_model = whisper_model if whisper_model else "base"
                    # Whisper might output to story.wav.json, so let's be explicit with the full path
                    whisper_command = [
                        'whisper', story_wav, 
                        '--model', current_whisper_model, 
                        '--output_format', 'json', 
                        '--output_dir', tmpdir, # Output dir is important
                        '--language', 'en', 
                        '--word_timestamps', 'True' 
                    ]
                    logger.info(f"Running Whisper command: {' '.join(whisper_command)}")
                    result = subprocess.run(whisper_command, capture_output=True, text=True, check=True)
                    
                    # Construct the expected JSON path based on Whisper's default naming (e.g., story.json)
                    expected_json_path = os.path.join(tmpdir, f"{Path(story_wav).stem}.json")

                    if not os.path.exists(expected_json_path) or os.path.getsize(expected_json_path) == 0:
                        logger.error(f"Whisper stdout: {result.stdout}")
                        logger.error(f"Whisper stderr: {result.stderr}")
                        raise Exception(f"No JSON file with word timestamps was generated by Whisper at {expected_json_path}. Check Whisper output for errors.")
                    
                    word_timestamps_json = expected_json_path # Update to the correct path
                    st.success("Word-level timestamps generated!")
                    
                except subprocess.CalledProcessError as e:
                    st.error(f"Whisper failed: {e.stderr}")
                    st.info("Ensure the Whisper model is downloaded and that the audio file is valid.")
                    st.stop()
                except Exception as e:
                    st.error(f"Word-level subtitle generation failed: {e}")
                    st.stop()

                # Step 4: Save uploaded video to a temporary file
                st.info("Saving uploaded video...")
                video_path = os.path.join(tmpdir, "background.mp4")
                try:
                    # Reset file pointer to beginning
                    video_file.seek(0)
                    with open(video_path, "wb") as f:
                        f.write(video_file.read())
                    if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
                        raise Exception(f"Video file was not saved correctly or is empty: {video_path}")
                    progress_bar.progress(75, text="Video saved to temporary folder.")
                except Exception as e:
                    st.error(f"Failed to save uploaded video: {e}")
                    st.stop()

                # Determine final video path
                if output_dir and os.path.isdir(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    final_video_output_path = os.path.join(output_dir, output_video_name)
                else:
                    # If output_dir is not provided or invalid, use tmpdir
                    final_video_output_path = os.path.join(tmpdir, output_video_name) 

                # Step 5: Merge audio and word-level subtitles with video using MoviePy
                st.info("Merging audio and word-level subtitles with video using MoviePy...")
                progress_bar.progress(80, text="Loading video, audio, and creating word clips...")

                video_clip = None
                audio_clip = None
                all_word_clips = [] # List to hold all individual word clips
                final_clip = None

                try:
                    # Load clips
                    video_clip = VideoFileClip(video_path)
                    audio_clip = AudioFileClip(story_wav)

                    # Read and parse the word-level JSON output from Whisper
                    with open(word_timestamps_json, 'r', encoding='utf-8') as f:
                        whisper_data = json.load(f)
                    
                    # Create individual ImageClips for each word using Pillow
                    for segment in whisper_data.get('segments', []):
                        for word_data in segment.get('words', []):
                            word_text = word_data['word'].strip() 
                            word_start = word_data['start']
                            word_end = word_data['end']

                            if word_text: 
                                # Render text with Pillow
                                pil_image = render_text_to_pillow_image(
                                    word_text, 
                                    font_path_for_moviepy, 
                                    font_size, 
                                    text_color, 
                                    stroke_color, 
                                    stroke_width, 
                                    bg_color_rgb, 
                                    bg_opacity, 
                                    padding,
                                    video_clip.w * 0.8 # Max 80% width of video for text
                                )
                                
                                # Convert Pillow image to a NumPy array
                                img_array = np.array(pil_image)
                                
                                # Convert NumPy array to MoviePy ImageClip
                                word_clip = ImageClip(img_array, ismask=False) 
                                
                                word_clip = word_clip.set_start(word_start).set_duration(word_end - word_start)
                                word_clip = word_clip.set_position(('center', 'center')) 

                                all_word_clips.append(word_clip)

                    # Calculate durations and adjust video length
                    audio_duration = audio_clip.duration
                    desired_duration = audio_duration + 1.0 # Add a small buffer at the end

                    if video_clip.duration < desired_duration:
                        loops = int(desired_duration // video_clip.duration) + 1
                        extended_video = video_clip.loop(n=loops).subclip(0, desired_duration)
                    else:
                        extended_video = video_clip.subclip(0, desired_duration)

                    # Composite all clips: extended video, its audio, and all individual word clips
                    final_clip = CompositeVideoClip([extended_video.set_audio(audio_clip)] + all_word_clips)
                    final_clip = final_clip.set_duration(desired_duration)

                    st.info(f"Writing final video to: {final_video_output_path}")
                    progress_bar.progress(90, text="Writing final video file...")
                    
                    final_clip.write_videofile(final_video_output_path, 
                                          codec='libx264', 
                                          audio_codec='aac',
                                          preset='medium',
                                          fps=video_clip.fps
                                         )

                    progress_bar.progress(100, text="Video created successfully!")
                    st.success("Video created successfully with MoviePy!")
                    
                    if os.path.exists(final_video_output_path) and os.path.getsize(final_video_output_path) > 0:
                        with open(final_video_output_path, "rb") as f:
                            st.download_button(
                                "Download Final Video", 
                                f.read(), 
                                file_name=output_video_name, 
                                mime="video/mp4",
                                key="download_final_video"
                            )
                        st.balloons()
                    else:
                        st.error(f"Final video file was not created or is empty at {final_video_output_path}")
                        
                except Exception as e:
                    st.error(f"An error occurred during video processing with MoviePy: {e}")
                    logger.error(f"MoviePy pipeline error: {e}", exc_info=True)
                    st.info("Check MoviePy's FFmpeg backend installation. Sometimes restarting Streamlit helps.")
                    st.stop()
                finally:
                    # Explicitly close all MoviePy objects to free resources
                    if video_clip: video_clip.close()
                    if audio_clip: audio_clip.close()
                    # It's good practice to close composite clips too
                    if final_clip: final_clip.close()
                    # Also close individual word clips if they're still in memory
                    for clip in all_word_clips:
                        if hasattr(clip, 'close'):
                            clip.close()

            except Exception as e:
                st.error(f"An unexpected error occurred in the overall pipeline: {e}")
                logger.error(f"Overall pipeline error: {e}", exc_info=True)

# --- Troubleshooting Tips ---
with st.expander("Troubleshooting Tips"):
    st.markdown("""
    **Common Issues and Solutions:**
    
    1.  **TTS "Kernel size" Error:**
        *   Ensure your story is at least 50 characters long after preprocessing.
        *   Try a different TTS model.
        *   Check that the story contains varied text and punctuation.
    
    2.  **Ollama Connection Issues:**
        *   Make sure Ollama is running: `ollama serve` in your terminal.
        *   Check if your model is available: `ollama list`
        *   Pull the model if needed: `ollama pull llama3` (or your chosen model).
    
    3.  **FFmpeg Issues (MoviePy dependency):**
        *   MoviePy requires FFmpeg. Ensure FFmpeg is installed and in your system's PATH.
        *   Try running `ffmpeg -version` in command prompt/terminal.
        *   Common fix: Download FFmpeg from [ffmpeg.org/download.html](https://ffmpeg.org/download.html).
        *   Extract to a simple path like `C:\\ffmpeg` (Windows) or `/usr/local/bin` (Linux/macOS) and add the `bin` directory to your system's PATH environment variable.
    
    4.  **Whisper Errors:**
        *   Ensure the Whisper model is downloaded. You can manually run `whisper --model [your_model_name]` (e.g., `whisper --model base`) once from your terminal to force a download.
        *   "No JSON file was generated": This might indicate a problem with Whisper's `--word_timestamps` feature or a general issue with Whisper processing the audio. Check the console output for specific errors.
    
    5.  **Memory Issues:**
        *   Using word-level subtitles can be more memory intensive. Use smaller models (e.g., "tiny", "base" for Whisper).
        *   Upload shorter background videos.
        *   Close other memory-intensive applications.
    
    6.  **Story Generation Issues:**
        *   Try different topics or prompts.
        *   Check Ollama model availability and ensure Ollama service is running.
    
    7.  **Video Processing Errors (MoviePy):**
        *   Check if the uploaded video file is corrupted or an unusual format. Try converting it to a standard MP4 outside the app first.
        *   Ensure sufficient disk space in your temporary directory or chosen output folder.
        *   Try shorter background videos (< 2 minutes).
        *   Sometimes, specific video codecs can cause issues. MoviePy generally handles common ones well, but highly compressed or obscure codecs might cause problems.

    8.  **Font Issues (If your custom font isn't showing):**
        *   **Check the Streamlit messages:** The app now provides more explicit messages about whether your uploaded font was successfully verified and is being used, or if it's falling back to the default. Look for `st.success` or `st.error` messages related to fonts during video generation.
        *   **Font File Validity:** Not all `.ttf` or `.otf` files are created equally. Some fonts might be malformed or use advanced features that Pillow doesn't fully support.
            *   **Try a very common font:** Download a well-known, simple font like `Roboto-Regular.ttf` (from Google Fonts) or `OpenSans-Regular.ttf` and try uploading that. If a common font works, the issue is likely with your specific custom font.
            *   **Corrupted file:** Ensure the font file itself isn't corrupted.
        *   **Verify in a text editor:** Can you open the font file with a standard font viewer or text editor on your operating system?
        *   **Permissions:** While unlikely in a temporary directory, ensure there are no strange file permissions preventing Pillow from reading the saved font file.
        *   **Pillow version:** Ensure your Pillow installation is up-to-date (`pip install --upgrade Pillow`).
        *   **Restart the app:** Sometimes a fresh restart of the Streamlit app can resolve caching or resource issues.
    """)

# Add debug option
with st.expander("Debug Information"):
    if st.button("Check Dependencies"):
        st.write("**Dependency Check:**")
        st.write(f"- FFmpeg available: {is_command_available('ffmpeg')}")
        st.write(f"- Whisper available: {is_command_available('whisper')}")
        st.write(f"- Ollama available: {is_command_available('ollama')}")
        st.write(f"- TTS available: {is_python_package_installed('TTS')}")
        st.write(f"- MoviePy available: {is_python_package_installed('moviepy')}")
        st.write(f"- Pillow available: {is_python_package_installed('PIL')}")
        st.write(f"- Numpy available: {is_python_package_installed('numpy')}")
        
        if is_command_available('ffmpeg'):
            try:
                result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
                st.write(f"- FFmpeg version: {result.stdout.splitlines()[0].split(' ')[2]}")
            except Exception as e:
                st.write(f"- FFmpeg version: Could not determine ({e})")
        
        if ollama_models:
            st.write(f"- Available Ollama models: {', '.join(ollama_models)}")
        else:
            st.write("- Available Ollama models: None (check if Ollama is running)")
