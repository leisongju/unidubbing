# Uni-Dubbing: Zero-Shot Speech Synthesis from Visual Articulation


## Setting Up the Environment
```bash
conda create -n unidubbing python=3.8
cd fairseq
git clone https://github.com/facebookresearch/fairseq/tree/afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb
pip install --editable ./
pip install -r requirements.txt
```

## Zero-Shot
### 1. Prepare a pretrained Hubert and HifiGAN (Acoustic unit)

Model | Pretraining Data                                                                               | Model | Quantizer
|---|------------------------------------------------------------------------------------------------|---|---
mHuBERT Base | En, Es, Fr speech | [download](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt) | [L11 km1000](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin)
HIFIGAN | 16k Universal                                                                                  | [download](https://zjueducn-my.sharepoint.com/:f:/g/personal/rongjiehuang_zju_edu_cn/EvMZ_WMcSoVDtUvE-C3wGhoBz4yI_N1Hcfk-LhzVnYMsvg?e=z59ntY)
dict.unit.txt|                                                   | [download](https://zjueducn-my.sharepoint.com/:t:/g/personal/rongjiehuang_zju_edu_cn/Ea5b_NwrBdNGlmNOun6V84sBGdAvFrl1ob2QrBwTYSDSYw?e=Rua4mN)

### Unit-to-Speech HiFi-GAN vocoder

Unit config | Unit size | Vocoder language | Dataset | Model
|---|---|---|---|---
mHuBERT, layer 11 | 1000 | En | [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json)
mHuBERT, layer 11 | 1000 | Es | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10/config.json)
mHuBERT, layer 11 | 1000 | Fr | [CSS10](https://github.com/Kyubyong/css10) | [ckpt](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/g_00500000), [config](https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_fr_css10/config.json)

```
python examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit \
  --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
  --results-path ${RESULTS_PATH} --dur-prediction
```

## Full-Shot
### 2.Prepare Semantic unit
Encode the audio using the pre-trained weights [HiFi-Codec-16k-320d](https://huggingface.co/Dongchao/AcademiCodec/blob/main/HiFi-Codec-16k-320d) from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec), and then decode the audio with the same model.



## Reference Repository
[Av-Hubert](https://github.com/facebookresearch/av_hubert)
[fairseq](https://github.com/facebookresearch/fairseq/tree/afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb)
[syncnet](https://github.com/joonson/syncnet_python)
[Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
[speech-resynthesis](https://github.com/facebookresearch/speech-resynthesis)
[AcademiCodec](https://github.com/yangdongchao/AcademiCodec)
[Transpeech](https://github.com/rongjiehuang/transpeech)