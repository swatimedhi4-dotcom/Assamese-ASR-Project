# Baseline: transcribe a clip with AI4Bharat IndicConformer (Assamese) using NeMo.
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.ASRModel.restore_from(
    "indicconformer_assamese.nemo",  # path to the downloaded IndicConformer model
    map_location="cpu"
)
result = model.transcribe(
    ["audio/clip01.wav"], language_id="as"  # 16 kHz mono WAV
)

print(result)
