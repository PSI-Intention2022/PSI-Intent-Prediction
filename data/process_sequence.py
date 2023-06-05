import numpy as np
import json
import os

def generate_data_sequence(set_name, database, args):
    intention_prob = []
    intention_binary = []
    frame_seq = []
    pids_seq = []
    video_seq = []
    box_seq = []
    description_seq = []
    disagree_score_seq = []

    video_ids = sorted(database.keys())
    for video in sorted(video_ids): # video_name: e.g., 'video_0001'
        for ped in sorted(database[video].keys()): # ped_id: e.g., 'track_1'
            frame_seq.append(database[video][ped]['frames'])
            box_seq.append(database[video][ped]['cv_annotations']['bbox'])

            n = len(database[video][ped]['frames'])
            pids_seq.append([ped] * n)
            video_seq.append([video] * n)
            intents, probs, disgrs, descripts = get_intent(database, video, ped, args)
            intention_prob.append(probs)
            intention_binary.append(intents)
            disagree_score_seq.append(disgrs)
            description_seq.append(descripts)

    return {
        'frame': frame_seq,
        'bbox': box_seq,
        'intention_prob': intention_prob,
        'intention_binary': intention_binary,
        'ped_id': pids_seq,
        'video_id': video_seq,
        'disagree_score': disagree_score_seq,
        'description': description_seq
    }


def get_intent(database, video_name, ped_id, args):
    prob_seq = []
    intent_seq = []
    disagree_seq = []
    description_seq = []
    n_frames = len(database[video_name][ped_id]['frames'])

    if args.intent_type == 'major' or args.intent_type == 'soft_vote':
        vid_uid_pairs = sorted((database[video_name][ped_id]['nlp_annotations'].keys()))
        n_users = len(vid_uid_pairs)
        for i in range(n_frames):
            labels = [database[video_name][ped_id]['nlp_annotations'][vid_uid]['intent'][i] for vid_uid in vid_uid_pairs]
            descriptions = [database[video_name][ped_id]['nlp_annotations'][vid_uid]['description'][i] for vid_uid in vid_uid_pairs]

            if args.intent_num == 3: # major 3 class, use cross-entropy loss
                uni_lbls, uni_cnts = np.unique(labels, return_counts=True)
                intent_binary = uni_lbls[np.argmax(uni_cnts)]
                if intent_binary == 'not_cross':
                    intent_binary = 0
                elif intent_binary == 'not_sure':
                    intent_binary = 1
                elif intent_binary == 'cross':
                    intent_binary = 2
                else:
                    raise Exception("ERROR intent label from database: ", intent_binary)

                intent_prob = np.max(uni_cnts) / n_users
                prob_seq.append(intent_prob)
                intent_seq.append(intent_binary)
                disagree_seq.append(1 - intent_prob)
                description_seq.append(descriptions)
            elif args.intent_num == 2: # only counts labels not "not-sure", but will involve issues if all annotators are not-sure.
                raise Exception("Sequence processing not implemented!")
            else:
                pass
    elif args.intent_type == 'mean':
        vid_uid_pairs = sorted((database[video_name][ped_id]['nlp_annotations'].keys()))
        n_users = len(vid_uid_pairs)
        for i in range(n_frames):
            labels = [database[video_name][ped_id]['nlp_annotations'][vid_uid]['intent'][i] for vid_uid in vid_uid_pairs]
            descriptions = [database[video_name][ped_id]['nlp_annotations'][vid_uid]['description'][i] for vid_uid in vid_uid_pairs]

            if args.intent_num == 2:
                for j in range(len(labels)):
                    if labels[j] == 'not_sure':
                        labels[j] = 0.5
                    elif labels[j] == 'not_cross':
                        labels[j] = 0
                    elif labels[j] == 'cross':
                        labels[j] = 1
                    else:
                        raise Exception("Unknown intent label: ", labels[j])
                # [0, 0.5, 1]
                intent_prob = np.mean(labels)
                intent_binary = 0 if intent_prob < 0.5 else 1
                prob_seq.append(intent_prob)
                intent_seq.append(intent_binary)
                disagree_score = sum([1 if lbl != intent_binary else 0 for lbl in labels]) / n_users
                disagree_seq.append(disagree_score)
                description_seq.append(descriptions)

    return intent_seq, prob_seq, disagree_seq, description_seq


