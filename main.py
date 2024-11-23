from torch import Tensor
import torchaudio

from pathlib import Path
from transformers import AutoProcessor, SeamlessM4Tv2Model


INPUT_MEDIA = Path(__file__).resolve().parent.joinpath("media/input")
OUTPUT_MEDIA = Path(__file__).resolve().parent.joinpath("media/output")

MODEL_NAME = "facebook/seamless-m4t-v2-large"
AUDIO_FORMAT = "wav"
DEFAULT_SAMPLING_RATE = 16_000
TRANSLATED_AUDIO = "{filename}_to_{target_language}{suffix}"


class Translator:
    def __init__(self, model_name: str) -> None:
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SeamlessM4Tv2Model.from_pretrained(model_name)
        self.sample_rate = self.model.config.sampling_rate

    def load_audio(self, path: Path) -> Tensor:
        audio, orig_freq = torchaudio.load(path)
        if orig_freq != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq,
                self.sample_rate,
            )
        return audio

    def translate_audio(self, audio: Tensor, target_language: str) -> Tensor:
        return self.model.generate(
            **self.processor(
                audios=audio,
                return_tensors="pt",
                sampling_rate=self.sample_rate,
            ),
            tgt_lang=target_language,
        )[
            0
        ].cpu()  # or .cuda()

    def save_audio(
        self,
        path: Path,
        audio: Tensor,
        format: str = AUDIO_FORMAT,
    ) -> None:
        torchaudio.save(
            uri=path,
            src=audio,
            sample_rate=self.sample_rate,
            format=format,
        )


if __name__ == "__main__":
    translator = Translator(MODEL_NAME)
    target_language = "eng"
    for file in INPUT_MEDIA.iterdir():
        translator.save_audio(
            OUTPUT_MEDIA.joinpath(
                TRANSLATED_AUDIO.format(
                    filename=file.name,
                    target_language=target_language,
                    suffix=file.suffix,
                ),
            ),
            translator.translate_audio(
                translator.load_audio(file),
                target_language,
            ),
        )
