# Sample Audio Files

This directory contains sample audio files for testing the audio transcription service.

## Test Audio Files

### 1. Clean Speech (`clean_speech.wav`)
- **Description**: Clear, high-quality speech without background noise
- **Duration**: 10 seconds
- **Expected Behavior**: 
  - Accurate transcription with high confidence
  - Clear word-level timestamps
  - Language detection should work well

### 2. Noisy Speech (`noisy_speech.wav`)
- **Description**: Speech with background noise (traffic, crowd, etc.)
- **Duration**: 15 seconds
- **Expected Behavior**:
  - Transcription accuracy may be lower
  - Vocal separation should improve results
  - VAD should help identify speech segments

### 3. Music with Vocals (`music_with_vocals.wav`)
- **Description**: Music track with vocals (song)
- **Duration**: 20 seconds
- **Expected Behavior**:
  - Vocal separation should extract vocals effectively
  - Transcription should focus on vocals, not instruments
  - Separation quality metrics should show improvement

### 4. Multi-Speaker (`multi_speaker.wav`)
- **Description**: Conversation between 2-3 speakers
- **Duration**: 30 seconds
- **Expected Behavior**:
  - Speaker diarization should identify different speakers
  - Each speaker should be labeled (SPEAKER_00, SPEAKER_01, etc.)
  - Speaker changes should be detected accurately

### 5. Very Noisy (`very_noisy.wav`)
- **Description**: Speech in very noisy environment (construction, machinery)
- **Duration**: 12 seconds
- **Expected Behavior**:
  - Graceful degradation in transcription quality
  - VAD should help identify speech vs noise
  - System should not crash or produce gibberish

### 6. Long Audio (`long_audio.wav`)
- **Description**: Extended audio file for testing chunking
- **Duration**: 2 minutes
- **Expected Behavior**:
  - Should trigger chunking for long audio
  - Chunks should be processed and stitched together
  - Performance should be reasonable

### 7. Different Languages (`spanish_speech.wav`)
- **Description**: Speech in Spanish
- **Duration**: 8 seconds
- **Expected Behavior**:
  - Language detection should identify Spanish
  - Transcription should be in Spanish
  - Language probability should be high

### 8. Low Quality (`low_quality.wav`)
- **Description**: Low bitrate, compressed audio
- **Duration**: 6 seconds
- **Expected Behavior**:
  - System should handle gracefully
  - Transcription quality may be reduced
  - Should not crash or produce errors

## Test Scenarios

### Basic Transcription
```bash
# Test clean speech
python -m src.cli transcribe sample_audio/clean_speech.wav

# Test with language hint
python -m src.cli transcribe sample_audio/spanish_speech.wav --language es
```

### Advanced Processing
```bash
# Test with vocal separation
python -m src.cli transcribe sample_audio/music_with_vocals.wav --enable-separation

# Test with speaker diarization
python -m src.cli transcribe sample_audio/multi_speaker.wav --enable-diarization --num-speakers 2

# Test with VAD
python -m src.cli transcribe sample_audio/noisy_speech.wav --enable-vad
```

### Batch Processing
```bash
# Process multiple files
python -m src.cli batch sample_audio/*.wav --output-dir results/
```

### API Testing
```bash
# Test via API
curl -X POST "http://localhost:8000/v1/transcribe" \
  -F "file=@sample_audio/clean_speech.wav" \
  -F 'config={"enable_separation": true, "language": "en"}'
```

## Expected Results

### Quality Metrics
- **Clean Speech**: >95% accuracy, >0.9 confidence
- **Noisy Speech**: >80% accuracy, >0.7 confidence
- **Music with Vocals**: >85% accuracy after separation
- **Multi-Speaker**: >90% accuracy, correct speaker identification
- **Very Noisy**: >60% accuracy, graceful degradation
- **Long Audio**: Successful chunking and stitching
- **Different Languages**: Correct language detection
- **Low Quality**: Handled gracefully, reduced quality

### Performance Targets
- **Processing Time**: <5 seconds for 30-second audio
- **Memory Usage**: <2GB RAM
- **Chunking**: Automatic for audio >30 seconds
- **Error Handling**: No crashes, informative error messages

## Test Data Generation

To generate additional test audio files:

```python
import numpy as np
import soundfile as sf

# Generate clean speech (sine wave)
sample_rate = 16000
duration = 10
t = np.linspace(0, duration, sample_rate * duration)
audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
sf.write('sample_audio/clean_speech.wav', audio, sample_rate)

# Generate noisy speech
noise = np.random.randn(len(audio)) * 0.1
noisy_audio = audio + noise
sf.write('sample_audio/noisy_speech.wav', noisy_audio, sample_rate)
```

## Notes

- All audio files should be in WAV format for consistency
- Sample rate should be 16kHz for optimal Whisper performance
- Files should be mono (single channel) unless testing stereo
- Keep file sizes reasonable (<10MB) for testing
- Include metadata about expected behavior in filenames

## Troubleshooting

### Common Issues
1. **File not found**: Ensure audio files are in the correct directory
2. **Format errors**: Check that files are valid WAV format
3. **Size limits**: Ensure files are under the maximum size limit
4. **Permission errors**: Check file permissions for reading

### Debug Mode
```bash
# Run with debug logging
python -m src.cli transcribe sample_audio/clean_speech.wav --debug

# Check service health
python -m src.cli health

# Get service capabilities
python -m src.cli capabilities
```
