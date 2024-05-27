# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
from itertools import chain
import logging
import math
import os
import sys
import json
import hashlib
import editdistance
from argparse import Namespace
import torch.nn as nn
import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.models import FairseqLanguageModel
from omegaconf import DictConfig

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    GenerationConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf
from sequence_generator import EnsembleModel




logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"

decode_type = 'word'


@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: [""], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})


@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def main(cfg: DictConfig):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for recognition!"
    assert (
            not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(cfg.common_eval.results_path, "decode.log")
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    return _main(cfg, sys.stdout, type=decode_type)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos, generator.pad}


def get_sent_res(cfg, num_sentences, gen_timer, result_dict):
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Recognized {:,} utterances ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))

    yaml_str = OmegaConf.to_yaml(cfg.generation)
    fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
    fid = fid % 1000000
    result_fn = f"{cfg.common_eval.results_path}/hypo-{fid}.json"
    json.dump(result_dict, open(result_fn, 'w'), indent=4)
    n_err, n_total = 0, 0
    assert len(result_dict['hypo']) == len(result_dict['ref'])
    for hypo, ref in zip(result_dict['hypo'], result_dict['ref']):
        hypo, ref = hypo.strip().split(), ref.strip().split()
        n_err += editdistance.eval(hypo, ref)
        n_total += len(ref)
    wer = 100 * n_err / n_total
    wer_fn = f"{cfg.common_eval.results_path}/wer.{fid}"
    with open(wer_fn, "w") as fo:
        fo.write(f"WER: {wer}\n")
        fo.write(f"err / num_ref_words = {n_err} / {n_total}\n\n")
        fo.write(f"{yaml_str}")
    logger.info(f"WER: {wer}%")
    return


def get_word_res(cfg, num_sentences, gen_timer, result_dict):
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Recognized {:,} utterances ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))

    yaml_str = OmegaConf.to_yaml(cfg.generation)
    fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
    fid = fid % 1000000
    result_fn = f"{cfg.common_eval.results_path}/hypo-{fid}.json"
    json.dump(result_dict, open(result_fn, 'w'), indent=4)
    n_acc, n_total = 0, 0
    assert len(result_dict['hypo']) == len(result_dict['ref'])
    for hypo, ref in zip(result_dict['hypo'], result_dict['ref']):
        hypo, ref = hypo.strip().split(), ref.strip().split()
        if hypo == ref:
            n_acc += 1
        n_total += 1
    acc = 100 * n_acc / n_total
    acc_fn = f"{cfg.common_eval.results_path}/acc.{fid}"
    with open(acc_fn, "w") as fo:
        fo.write(f"ACC: {acc}\n")
        fo.write(f"acc / num_refs = {n_acc} / {n_total}\n\n")
        fo.write(f"{yaml_str}")
    logger.info(f"acc: {acc}%")
    return


def _main(cfg, output_file, type='sent'):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("hybrid.speech_recognize")
    if output_file is not sys.stdout:  # also print to stdout
        logger.addHandler(logging.StreamHandler(sys.stdout))

    import torch

    def decode_fn(x):
        # 将每个元素减去4
        x_subtracted = x - 4
        
        x_list = x_subtracted.tolist()
        # 将结果转换为整数，然后转换为字符串
        x_str = [str(int(item)) for item in x_list]

        # 将字符串连接起来，中间用空格分隔
        result = ' '.join(x_str)

        return result


    utils.import_user_module(cfg.common)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path])
    print(saved_cfg)
    models = [model.eval().cuda() for model in models]
    # saved_cfg.task.labels = ["wrd"]
    task = tasks.setup_task(saved_cfg.task)

    task.build_tokenizer(saved_cfg.tokenizer)
    task.build_bpe(saved_cfg.bpe)

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available()

    # Set dictionary
    # dictionary = task.target_dictionary

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav
    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir
    if cfg.override.modalities is not None:
        task.cfg.modalities = cfg.override.modalities

    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        # model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )



    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    result_dict = {'utt_id': [], 'ref': [], 'hypo': []}
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample

        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        # if sample['net_input']['source']['audio'] is not None:
        #     sample['net_input']['source']['audio'] = None
        net_output = models[0](sample['net_input']['source'],sample["target_list"],sample['net_input']['padding_mask'])
        logp_u_list, targ_u_list = net_output['logit_u_list'], net_output['target_u_list']
        
        hypos = logp_u_list[0].argmax(dim=-1).int().cpu()
        start, end = 0, 0
        for i in range(len(sample["id"])):
            result_dict['utt_id'].append(sample['utt_id'][i])
            # index = sample['target_list'][0][i].shape

            ref_sent = decode_fn(sample['target_list'][0][i].int().cpu())
            result_dict['ref'].append(ref_sent)
            end = start + len(ref_sent.split(' '))
            best_hypo = hypos[start:end]
            start = end
            hypo_str = decode_fn(best_hypo)
            # print("target %d, hypo %d" % (len(ref_sent.split(' ')), len(hypo_str.split(' '))))
            result_dict['hypo'].append(hypo_str)

    n_err, n_total = 0, 0
    assert len(result_dict['hypo']) == len(result_dict['ref'])
    for hypo, ref in zip(result_dict['hypo'], result_dict['ref']):
        hypo, ref = hypo.strip().split(), ref.strip().split()
        n_err += editdistance.eval(hypo, ref)
        n_total += len(ref)
    wer = 100 * n_err / n_total
    wer_fn = f"../../../../asr/predict_unit/wer.unit"
    with open(wer_fn, "w") as fo:
        fo.write(f"WER: {wer}\n")
        fo.write(f"err / num_ref_words = {n_err} / {n_total}\n\n")
    logger.info(f"WER: {wer}%")
    result_fn = f"{cfg.common_eval.results_path}/hypo.json"
    json.dump(result_dict, open(result_fn, 'w'), indent=4)
    experimental = cfg.common_eval.path.split('/')[-3]
    with open('../../../../asr/result.csv', 'a') as file:
        file.write(f"\nexperimental: {experimental}\tUnit WER: {wer}\n")



@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

    wer = float("inf")

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))
    return


def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cli_main()

    """
    python -B infer_s2s.py \
        --config-dir ./conf \
        --config-name s2s_decode.yaml \
        dataset.gen_subset=test \
        common_eval.results_path=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/project/openLR/av_hubert/res\
        override.modalities=['audio'] common.user_dir=`pwd`\
        override.data=/mnt/disk2/chengxize/data/tedxMultilingual/30h \
        override.label_dir=/mnt/disk2/chengxize/data/tedxMultilingual/30h \
        common_eval.path=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/project/openLR/av_hubert/finetune/base_lrs3_30h.pt

    python -B infer_s2s.py \
        --config-dir ./conf \
        --config-name s2s_decode.yaml \
        dataset.gen_subset=test \
        common_eval.results_path=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/project/openLR/av_hubert/res\
        override.modalities=['video'] common.user_dir=`pwd`\
        override.data=/mnt/disk1/chengxize/data/lrs3/30h_data\
        override.label_dir=/mnt/disk1/chengxize/data/lrs3/30h_data \
        common_eval.path=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/project/openLR/av_hubert/model/large_vox_video_openlr/checkpoints/checkpoint_best.pt

    python -B infer_s2s.py \
        --config-dir ./conf \
        --config-name s2s_decode.yaml \
        dataset.gen_subset=test \
        common_eval.results_path=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/project/openLR/av_hubert/res\
        override.modalities=['audio','video'] common.user_dir=`pwd`\
        override.data=/mnt/disk1/chengxize/project/avtrans/res/covost_cvss_face_ori \
        override.label_dir=/mnt/disk1/chengxize/project/avtrans/res/covost_cvss_face_ori \
        common_eval.path=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/project/openLR/av_hubert/lrs3_model/finetune_openlr_large_vox_433h_audio/checkpoints/checkpoint_best.pt
    """

    """
    python -B infer_s2s.py \
        --config-dir ./conf \
        --config-name s2s_decode.yaml \
        dataset.gen_subset=test \
        common_eval.results_path=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/project/openLR/av_hubert/res\
        override.modalities=['video'] common.user_dir=`pwd`\
        override.data=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/data/lrs2/30h_data\
        override.label_dir=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/data/lrs2/30h_data \
        common_eval.path=/mnt/disk1/chengxize/project/openLR/av_hubert/prompt/
    """

    """
    python -B infer_s2s.py \
            --config-dir ./conf \
            --config-name s2s_decode.yaml \
            dataset.gen_subset=test \
            common_eval.results_path=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/project/openLR/av_hubert/res\
            override.modalities=['video'] common.user_dir=`pwd`\
            override.data=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/data/CMLR/all_data\
            override.label_dir=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/data/CMLR/all_data\
            common_eval.path=/mnt/e5254a2d-db6d-420a-b4ea-ee215b9c32a3/chengxize/project/openLR/av_hubert/model/large_vox_video_openlr/checkpoints/checkpoint_best.pt
    """
