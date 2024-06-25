import streamlit as st
import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from IPython.display import Audio
from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
from io import BytesIO
import soundfile as sf
import tempfile

# Load Pretrained Model with Hyperparameters from Source Folder
@st.cache_resource
def load_pretrained_separator():
    source_folder = "sepformer-dns"  # Your source folder
    model = separator.from_hparams(
        source=source_folder,
        savedir=source_folder,
        run_opts={"device": "cpu"}  # Adjust for GPU if available
    )
    return model

# Initialize the Model
model = load_pretrained_separator()

# Title
st.title("Speech Enhancement with SpeechBrain")

# Input Options
input_type = st.radio("Input Audio:", ("Upload .wav file", "Record Audio"))

audio_data = None
sample_rate = None  # Initialize sample_rate variable
if input_type == "Upload .wav file":
    audio_file = st.file_uploader("Choose a .wav file", type="wav")
    if audio_file:
        audio_data, sample_rate = sf.read(audio_file)  # Read audio and sample rate

elif input_type == "Record Audio":
    audio_bytes = st.audio_recorder("Record your audio")
    if audio_bytes:
        temp_file = 'audio_cache/audio.wav'
        temp_file.write(audio_bytes)
        temp_file.close()
        audio_data, sample_rate = sf.read(temp_file.name) 
        

# Enhancement and Playback/Download
if audio_data is not None and sample_rate is not None:
    enhanced_speech=None
    with st.spinner("Enhancing audio..."):
        
        audiopath='audio_cache/audio.wav'
        # Use the retrieved sample rate
        print('Enhancing...')
        sf.write(audiopath, audio_data, samplerate=sample_rate)  
        enhanced_speech = model.separate_file(path=audiopath)
        
    # Playback
    if enhanced_speech is not None:
        sample_rate=16000
        enhanced_audio = enhanced_speech[:, :].detach().cpu().squeeze()
        # Save to Temporary File
        enhancedaudiopath='audio_cache/enhancedaudio.wav'
        # Save enhanced audio to bytes buffer
        sf.write(enhancedaudiopath, enhanced_audio.numpy(), samplerate=sample_rate, format='WAV')
        if enhancedaudiopath:
            try:
                # Read audio data and sample rate
                audio_data, sample_rate = sf.read(enhancedaudiopath)

                # Display audio player in Streamlit
                st.audio(audio_data, format="audio/wav",sample_rate=sample_rate) 
            except FileNotFoundError:
                st.error("File not found at the specified path.")
            except Exception as e:
                st.error(f"Error reading or playing audio: {e}")


        # Download
        with open(enhancedaudiopath, "rb") as f:
            st.download_button(
                label="Download Enhanced Audio", 
                data=f, 
                file_name="enhanced_audio.wav", 
                mime="audio/wav"
            )