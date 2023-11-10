from pesq import pesq
from pystoi import stoi
import numpy
import os
import pickle
from tqdm import tqdm
import torch
import torchaudio
import pandas as pd
import numpy as np
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import matplotlib.pyplot as plt
import torchaudio.functional as F

print("np_ver:", numpy.version.version)

noisy_dir = "/data/ephraim/datasets_16k/noisy_testset_wav"
clean_dir = "/data/ephraim/datasets_16k/clean_testset_wav"
enhance_dir = "/data/ephraim/output/Enhanced/pretrained_base/model370200/test/voicebank_Noisy_Test"

references = os.listdir(clean_dir)
pkl_results_file = "SE_measures_pretrained_generic_obj_all.pickle"

objective_model = SQUIM_OBJECTIVE.get_model()
subjective_model = SQUIM_SUBJECTIVE.get_model()

NMR_SPEECH = os.path.join(clean_dir, "p232_001.wav")
WAVEFORM_NMR, SAMPLE_RATE_NMR = torchaudio.load(NMR_SPEECH)
if SAMPLE_RATE_NMR != 16000:
    WAVEFORM_NMR = F.resample(WAVEFORM_NMR, SAMPLE_RATE_NMR, 16000)


def calc_measures():
    dont_calculated = []
    results = {
        "pesq_noisy": {},
        "stoi_noisy": {},
        "stoi_est_noisy": {},
        "pesq_est": {},
        "sisdr_est_noisy": {},
        "pesq_enhanced": {},
        "stoi_enhanced": {},
        "stoi_est_enhanced": {},
        "pesq_est_enhanced": {},
        "sisdr_est_enhanced": {},
        "mos_est_noise": {},
        "mos_est_enhanced": {}
    }

    i = 0

    for ref_filename in tqdm(references):

        reference = os.path.join(clean_dir, ref_filename)
        test_noisy = os.path.join(noisy_dir, ref_filename)
        test_enhanced = os.path.join(enhance_dir, ref_filename)
        WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH = torchaudio.load(reference)
        WAVEFORM_NOISE, SAMPLE_RATE_NOISE = torchaudio.load(test_noisy)
        WAVEFORM_enhanced, SAMPLE_RATE_enhanced = torchaudio.load(test_enhanced)
        print("Computing scores for ", reference)
        try:
            pesq_noise = pesq(
                16000,
                WAVEFORM_SPEECH[0].numpy(),
                WAVEFORM_NOISE[0].numpy(),
                mode="wb",
            )
            stoi_noise = stoi(
                WAVEFORM_SPEECH[0].numpy(),
                WAVEFORM_NOISE[0].numpy(),
                16000,
                extended=False,
            )
            pesq_enhanced = pesq(
                16000,
                WAVEFORM_SPEECH[0].numpy(),
                WAVEFORM_enhanced[0].numpy(),
                mode="wb",
            )
            stoi_enhanced = stoi(
                WAVEFORM_SPEECH[0].numpy(),
                WAVEFORM_enhanced[0].numpy(),
                16000,
                extended=False,
            )
            stoi_est_noise, pesq_est_noise, sisdr_est_noise = objective_model(
                WAVEFORM_NOISE[0:1, :]
            )
            stoi_est_enhanced, pesq_est_enhanced, sisdr_est_enhanced = objective_model(
                WAVEFORM_enhanced[0:1, :]
            )
            mos_est_noise = subjective_model(WAVEFORM_NOISE[0:1, :], WAVEFORM_NMR)
            mos_est_enhanced = subjective_model(WAVEFORM_enhanced[0:1, :], WAVEFORM_NMR)

            results["pesq_noisy"][ref_filename] = pesq_noise
            results["stoi_noisy"][ref_filename] = stoi_noise
            results["stoi_est_noisy"][ref_filename] = float(stoi_est_noise)
            results["pesq_est_noisy"][ref_filename] = float(pesq_est_noise)
            results["sisdr_est_noisy"][ref_filename] = float(sisdr_est_noise)
            results["pesq_enhanced"][ref_filename] = pesq_enhanced
            results["stoi_enhanced"][ref_filename] = stoi_enhanced
            results["pesq_enhanced"][ref_filename] = pesq_enhanced
            results["stoi_enhanced"][ref_filename] = stoi_enhanced
            results["stoi_est_enhanced"][ref_filename] = float(stoi_est_enhanced)
            results["pesq_est_enhanced"][ref_filename] = float(pesq_est_enhanced)
            results["sisdr_est_enhanced"][ref_filename] = float(sisdr_est_enhanced)
            results["mos_est_noise"][ref_filename] =  float(mos_est_noise)
            results["mos_est_enhanced"][ref_filename] =  float(mos_est_enhanced)
        except:
            dont_calculated.append(ref_filename)
    return results


if os.path.exists(pkl_results_file):
    with open(pkl_results_file, "rb") as handle:
        results = pickle.load(handle)
else:
    results = calc_measures()
    with open(pkl_results_file, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


# new_dict = {
#     "pesq_noisy": {},
#     "pesq_enhanced": {},
#     "stoi_noisy": {},
#     "stoi_enhanced": {},
# }
# for filename, file_data in results.items():
#     # for noisy_enhanced, data_dic in file_data.items():
#     new_dict["pesq_noisy"][filename] = np.mean(file_data["noisy"]["pesq"])
#     new_dict["pesq_enhanced"][filename] = np.mean(file_data["enhanced"]["pesq"])
#     new_dict["stoi_noisy"][filename] = np.mean(file_data["noisy"]["stoi"])
#     new_dict["stoi_enhanced"][filename] = np.mean(file_data["enhanced"]["stoi"])

df = pd.DataFrame.from_dict(results)
df["pesq_diff"] = df["pesq_enhanced"].sub(df["pesq_noisy"])
df["stoi_diff"] = df["stoi_enhanced"].sub(df["stoi_noisy"])
df["stoi_est_diff"] = df["stoi_est_enhanced"].sub(df["stoi_est_noisy"])
df["pesq_est_diff"] = df["pesq_est_enhanced"].sub(df["pesq_est"])
df["sisdr_est_est_diff"] = df["sisdr_est_enhanced"].sub(df["sisdr_est_noisy"])
df["mos_est_diff"] = df["mos_est_enhanced"].sub(df["mos_est_noise"])

print(df.describe())
