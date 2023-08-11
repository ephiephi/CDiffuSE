# -*- coding: utf-8 -*-
"""
Data preparation.

Download and resample, use ``download_vctk`` below.
https://datashare.is.ed.ac.uk/handle/10283/2791

Authors:
 * Szu-Wei Fu, 2020
 * Peter Plantinga, 2020
"""

import os
import json

import shutil
# import logging
# import tempfile
import torchaudio
from torchaudio.transforms import Resample
# logger = logging.getLogger(__name__)
LEXICON_URL = "http://www.openslr.org/resources/11/librispeech-lexicon.txt"
TRAIN_JSON = "train.json"
TEST_JSON = "test.json"
VALID_JSON = "valid.json"
SAMPLERATE = 16000
TRAIN_SPEAKERS = [
    "p226",
    "p287",
    "p227",
    "p228",
    "p230",
    "p231",
    "p233",
    "p236",
    "p239",
    "p243",
    "p244",
    "p250",
    "p254",
    "p256",
    "p258",
    "p259",
    "p267",
    "p268",
    "p269",
    "p270",
    "p273",
    "p274",
    "p276",
    "p277",
    "p278",
    "p279",
    "p282",
    "p286",
]



# def create_json(wav_lst, json_file, clean_folder, txt_folder, lexicon):
#     """
#     Creates the json file given a list of wav files.

#     Arguments
#     ---------
#     wav_lst : list
#         The list of wav files.
#     json_file : str
#         The path of the output json file
#     clean_folder : str
#         The location of parallel clean samples.
#     txt_folder : str
#         The location of the transcript files.
#     """
#     logger.debug(f"Creating json lists in {json_file}")

#     # Processing all the wav files in the list
#     json_dict = {}
#     for wav_file in wav_lst:  # ex:p203_122.wav

#         # Example wav_file: p232_001.wav
#         noisy_path, filename = os.path.split(wav_file)
#         _, noisy_dir = os.path.split(noisy_path)
#         _, clean_dir = os.path.split(clean_folder)
#         noisy_rel_path = os.path.join("{data_root}", noisy_dir, filename)
#         clean_rel_path = os.path.join("{data_root}", clean_dir, filename)

#         # Reading the signal (to retrieve duration in seconds)
#         signal = read_audio(wav_file)
#         duration = signal.shape[0] / SAMPLERATE

#         # Read text
#         snt_id = filename.replace(".wav", "")
#         with open(os.path.join(txt_folder, snt_id + ".txt")) as f:
#             word_string = f.read()
#         word_string = remove_punctuation(word_string).strip().upper()
#         phones = [
#             phn for word in word_string.split() for phn in lexicon[word].split()
#         ]

#         # Remove duplicate phones
#         phones = [i for i, j in zip(phones, phones[1:] + [None]) if i != j]
#         phone_string = " ".join(phones)

#         json_dict[snt_id] = {
#             "noisy_wav": noisy_rel_path,
#             "clean_wav": clean_rel_path,
#             "length": duration,
#             "words": word_string,
#             "phones": phone_string,
#         }

#     # Writing the json lines
#     with open(json_file, mode="w") as json_f:
#         json.dump(json_dict, json_f, indent=2)

#     logger.info(f"{json_file} successfully created!")


# def check_voicebank_folders(*folders):
    # """Raises FileNotFoundError if any passed folder does not exist."""
    # for folder in folders:
    #     if not os.path.exists(folder):
    #         raise FileNotFoundError(
    #             f"the folder {folder} does not exist (it is expected in "
    #             "the Voicebank dataset)"
    #         )

def get_all_files(
    dirName, match_and=None, match_or=None, exclude_and=None, exclude_or=None
):
    """Returns a list of files found within a folder.

    Different options can be used to restrict the search to some specific
    patterns.

    Arguments
    ---------
    dirName : str
        The directory to search.
    match_and : list
        A list that contains patterns to match. The file is
        returned if it matches all the entries in `match_and`.
    match_or : list
        A list that contains patterns to match. The file is
        returned if it matches one or more of the entries in `match_or`.
    exclude_and : list
        A list that contains patterns to match. The file is
        returned if it matches none of the entries in `exclude_and`.
    exclude_or : list
        A list that contains pattern to match. The file is
        returned if it fails to match one of the entries in `exclude_or`.

    Example
    -------
    >>> get_all_files('tests/samples/RIRs', match_and=['3.wav'])
    ['tests/samples/RIRs/rir3.wav']
    """

    # Match/exclude variable initialization
    match_and_entry = True
    match_or_entry = True
    exclude_or_entry = False
    exclude_and_entry = False

    # Create a list of file and sub directories
    listOfFile = os.listdir(dirName)
    allFiles = list()

    # Iterate over all the entries
    for entry in listOfFile:

        # Create full path
        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files(
                fullPath,
                match_and=match_and,
                match_or=match_or,
                exclude_and=exclude_and,
                exclude_or=exclude_or,
            )
        else:

            # Check match_and case
            if match_and is not None:
                match_and_entry = False
                match_found = 0

                for ele in match_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(match_and):
                    match_and_entry = True

            # Check match_or case
            if match_or is not None:
                match_or_entry = False
                for ele in match_or:
                    if ele in fullPath:
                        match_or_entry = True
                        break

            # Check exclude_and case
            if exclude_and is not None:
                match_found = 0

                for ele in exclude_and:
                    if ele in fullPath:
                        match_found = match_found + 1
                if match_found == len(exclude_and):
                    exclude_and_entry = True

            # Check exclude_or case
            if exclude_or is not None:
                exclude_or_entry = False
                for ele in exclude_or:
                    if ele in fullPath:
                        exclude_or_entry = True
                        break

            # If needed, append the current file to the output list
            if (
                match_and_entry
                and match_or_entry
                and not (exclude_and_entry)
                and not (exclude_or_entry)
            ):
                allFiles.append(fullPath)

    return allFiles


def download_vctk(destination, tmp_dir=None, device="cpu"):
    """Download dataset and perform resample to 16000 Hz.

    Arguments
    ---------
    destination : str
        Place to put final dataset.
    tmp_dir : str
        Location to store temporary files. Will use `tempfile` if not provided.
    device : str
        Passed directly to pytorch's ``.to()`` method. Used for resampling.
    """
    dataset_name = "noisy-vctk-16k"

    # final_dir = os.path.join(tmp_dir, dataset_name)
    final_dir = destination

    # if not os.path.isdir(tmp_dir): #tempdir - origin
    #     os.mkdir(tmp_dir)

    if not os.path.isdir(final_dir):
        os.mkdir(final_dir)

    # prefix = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/"

    # Move transcripts to final dir
    # shutil.move(os.path.join(tmp_dir, "testset_txt"), final_dir)
    # shutil.move(os.path.join(tmp_dir, "trainset_28spk_txt"), final_dir)

    # Downsample
    dirs = [
        "noisy_testset_wav",
        "clean_testset_wav",
        "noisy_trainset_28spk_wav",
        "clean_trainset_28spk_wav",
    ]

    downsampler = Resample(orig_freq=48000, new_freq=16000)

    for directory in dirs:
        print("Resampling " + directory)
        dirname = os.path.join(tmp_dir, directory)

        # Make directory to store downsampled files
        dirname_16k = os.path.join(final_dir, directory + "_16k")
        if not os.path.isdir(dirname_16k):
            os.mkdir(dirname_16k)

        # Load files and downsample
        for filename in get_all_files(dirname, match_and=[".wav"]):
            signal, rate = torchaudio.load(filename)
            downsampled_signal = downsampler(signal.view(1, -1).to(device))

            # Save downsampled file
            torchaudio.save(
                os.path.join(dirname_16k, filename[-12:]),
                downsampled_signal.cpu(),
                sample_rate=16000,
            )

            # Remove old file
            os.remove(filename)

        # Remove old directory
        os.rmdir(dirname)

    # logger.info("Zipping " + final_dir)
    # final_zip = shutil.make_archive(
    #     base_name=final_dir,
    #     format="zip",
    #     root_dir=os.path.dirname(final_dir),
    #     base_dir=os.path.basename(final_dir),
    # )

    # logger.info(f"Moving {final_zip} to {destination}")
    # shutil.move(final_zip, os.path.join(destination, dataset_name + ".zip"))
 
download_vctk(destination="/data/ephraim/detasets_16k/", tmp_dir="/data/ephraim/datasets_temp/", device="cpu")