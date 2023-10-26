import speechmetrics as sm
import pprint
import numpy
import os
import pickle
from tqdm import tqdm

print("np_ver:", numpy.version.version)

noisy_dir = "/data/ephraim/datasets_16k/noisy_testset_wav"
clean_dir = "/data/ephraim/datasets_16k/clean_testset_wav"
enhance_dir = "/data/ephraim/output/Enhanced/second_train/model37800/test/voicebank_Noisy_Test/"

metrics = sm.load(["pesq", "stoi"])
references = os.listdir(clean_dir)
pkl_results_file = "SE_measures.pickle"


def calc_measures():
    dont_calculated = []
    results = {}
    i=0
    
    for ref_filename in tqdm(references):
        reference = os.path.join(clean_dir, ref_filename)
        test_noisy = os.path.join(noisy_dir, ref_filename)
        test_enhanced = os.path.join(enhance_dir, ref_filename)
        print('Computing scores for ', reference)
        try:
            scores_noisy = metrics(reference, test_noisy)
            scores_enhanced = metrics(reference, test_enhanced)
            results[ref_filename]= {"noisy": scores_noisy, "enhanced": scores_enhanced}
        except:
            dont_calculated.append(ref_filename)
    return results
        



if os.path.exists(pkl_results_file):
    with open(pkl_results_file, 'rb') as handle:
        results = pickle.load(handle)
else:
    results = calc_measures()
    with open(pkl_results_file, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


print(results)













