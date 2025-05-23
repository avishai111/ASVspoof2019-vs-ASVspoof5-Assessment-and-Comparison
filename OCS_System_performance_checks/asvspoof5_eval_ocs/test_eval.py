import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dataset import ASVspoof2019, ASVspoof5
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import eval_metrics as em
import numpy as np
torch.nn.Module.dump_patches = True

def compute_eer(cm_score_file, path_to_database):
    #asv_score_file = os.path.join(path_to_database, 'LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 2]
    cm_keys = cm_data[:, 3]
    cm_scores = cm_data[:, 4].astype(np.float64)
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]
    other_cm_scores = -cm_scores
    other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_keys == 'spoof'])[0]
    if eer_cm < other_eer_cm:
        print("EER:",eer_cm)
        return eer_cm
    else:
        print("EER:",other_eer_cm)  
        return other_eer_cm

def test_model(feat_model_path, loss_model_path, part, add_loss, device,genuine_only,asvspoof2019 = False):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    model = torch.load(feat_model_path, map_location="cuda")
    model = model.to(device)
    loss_model = torch.load(loss_model_path) if add_loss != "softmax" else None
    if asvspoof2019:
        test_set = ASVspoof2019("LA", "E:/AIR-ASVspoof-master/LA/Features/",
                                "E:/AIR-ASVspoof-master/LA/ASVspoof2019_LA_cm_protocols/", part,
                                "LFCC", feat_len=750, padding="repeat",include_gender=True, genuine_only = genuine_only)
        dataset_name = "ASVspoof2019"
    else:
        test_set = ASVspoof5("LA", "F:/ASVSpoof5/Features_pkl/",
                                "E:/ASVSpoof5/cm_protocols/converted_ASVSpoof5/", part,
                                "LFCC", feat_len=750, padding="repeat",include_gender=True, genuine_only = genuine_only)
        dataset_name = "ASVspoof5"
    
    testDataLoader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0,
                                collate_fn=test_set.collate_fn)
    model.eval()

    if genuine_only:
        file_name = f'{dataset_name}_bonafide/checkpoint_cm_score_gender_{part}_bonafide.txt'
    else:
        file_name = f'{dataset_name}_bonafide/checkpoint_cm_score_gender_{part}.txt'
    
    with open(os.path.join(dir_path, file_name), 'w') as cm_score_file:
        for i, (lfcc, audio_fn, tags, labels,gender, speaker) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.unsqueeze(1).float().to(device)
            tags = tags.to(device)
            labels = labels.to(device)

            feats, lfcc_outputs = model(lfcc)

            score = F.softmax(lfcc_outputs)[:, 0]

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]

            for j in range(labels.size(0)):
                cm_score_file.write(
                    '%s %s A%02d %s %s %s\n' % (audio_fn[j], speaker[j] ,tags[j].data,
                                          "spoof" if labels[j].data.cpu().numpy() else "bonafide",
                                          score[j].item(),gender[j]))
    eer_cm = compute_eer(os.path.join(dir_path, file_name),"E:/AIR-ASVspoof-master/")
    print("The eer cm is:",eer_cm)
    with open(os.path.join(dir_path, f"{dataset_name}_{part}_cm_eer.txt"),  "w") as file:
        # Write the output to the file
        file.write(f"The eer cm is: {eer_cm}\n")
    # eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(dir_path, 'checkpoint_cm_score.txt'),
    #                                          "E:/AIR-ASVspoof-master/")
    return eer_cm

def test(model_dir, add_loss, device):
    model_path = os.path.join(model_dir, "anti-spoofing_lfcc_model.pt")
    loss_model_path = os.path.join(model_dir, "anti-spoofing_loss_model.pt")
    #test_model(model_path, loss_model_path, "eval", add_loss, device)
    # test_model(model_path, loss_model_path, "dev", add_loss, device,genuine_only = True,asvspoof2019 = True)
    # test_model(model_path, loss_model_path, "dev", add_loss, device,genuine_only = False,asvspoof2019 = True)
    # test_model(model_path, loss_model_path, "train", add_loss, device,genuine_only =True,asvspoof2019 = True)
    # test_model(model_path, loss_model_path, "train", add_loss, device,genuine_only =False,asvspoof2019 = True)
    # test_model(model_path, loss_model_path, "eval", add_loss, device,genuine_only =True,asvspoof2019 = True)
    # test_model(model_path, loss_model_path, "eval", add_loss, device,genuine_only = False,asvspoof2019 = True)
    
    # test_model(model_path, loss_model_path, "dev", add_loss, device,genuine_only = True,asvspoof2019 = False)
    test_model(model_path, loss_model_path, "dev", add_loss, device,genuine_only = False,asvspoof2019 = False)
    test_model(model_path, loss_model_path, "train", add_loss, device,genuine_only =True,asvspoof2019 = False)
    test_model(model_path, loss_model_path, "train", add_loss, device,genuine_only =False,asvspoof2019 = False)
    test_model(model_path, loss_model_path, "eval", add_loss, device,genuine_only =True,asvspoof2019 = False)
    test_model(model_path, loss_model_path, "eval", add_loss, device,genuine_only = False,asvspoof2019 = False)
   

def test_individual_attacks(cm_score_file):
    asv_score_file = os.path.join('/data/neil/DS_10283_3336',
                                  'LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float)

    other_cm_scores = -cm_scores

    eer_cm_lst, min_tDCF_lst = [], []
    for attack_idx in range(7,20):
        # Extract target, nontarget, and spoof scores from the ASV scores
        tar_asv = asv_scores[asv_keys == 'target']
        non_asv = asv_scores[asv_keys == 'nontarget']
        spoof_asv = asv_scores[asv_sources == 'A%02d' % attack_idx]

        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_sources == 'A%02d' % attack_idx]

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
        eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

        other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_sources == 'A%02d' % attack_idx])[0]

        [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

        if eer_cm < other_eer_cm:
            # Compute t-DCF
            tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model,
                                                        True)
            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]

        else:
            tDCF_curve, CM_thresholds = em.compute_tDCF(other_cm_scores[cm_keys == 'bonafide'],
                                                        other_cm_scores[cm_sources == 'A%02d' % attack_idx],
                                                        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)
            # Minimum t-DCF
            min_tDCF_index = np.argmin(tDCF_curve)
            min_tDCF = tDCF_curve[min_tDCF_index]
        eer_cm_lst.append(min(eer_cm, other_eer_cm))
        min_tDCF_lst.append(min_tDCF)

    return eer_cm_lst, min_tDCF_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model_dir', type=str, help="path to the trained model", default="./AIR-ASVspoof-master/models1028/ocsoftmax/")
    parser.add_argument('-l', '--loss', type=str, default="ocsoftmax",
                        choices=["softmax", 'amsoftmax', 'ocsoftmax'], help="loss function")
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(args.model_dir, args.loss, args.device)
    # eer_cm_lst, min_tDCF_lst = test_individual_attacks(os.path.join(args.model_dir, 'checkpoint_cm_score.txt'))
    # print(eer_cm_lst)
    # print(min_tDCF_lst)
