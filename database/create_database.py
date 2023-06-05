import json
import os
import time
import pickle

'''
Database organization

db = {
    'video_name': {
        'pedestrian_id': # track_id-#
        { 
            'frames': [0, 1, 2, ...], # target pedestrian appeared frames
            'cv_annotations': {
                'track_id': track_id, 
                'bbox': [xtl, ytl, xbr, ybr], 
            },
            'nlp_annotations': {
                vid_uid_pair: {'intent': [], 'description': [], 'key_frame': []},
                ...
            }
        }
    }
}
'''


def create_database(args):
    for split_name in ['train', 'val', 'test']:
        with open(args.video_splits) as f:
            datasplits = json.load(f)
        db_log = os.path.join(args.database_path, split_name + '_db_log.txt')
        with open(db_log, 'w') as f:
            f.write(f"Initialize {split_name} database \n")
            f.write(time.strftime("%d%b%Y-%Hh%Mm%Ss") + "\n")
        # 1. Init db
        db = init_db(sorted(datasplits[split_name]), db_log, args)
        # 2. get intent, remove missing frames
        update_db_annotations(db, db_log, args)
        # 3. cut sequences, remove early frames before the first key frame, and after last key frame
        # cut_sequence(db, db_log, args)

        database_name = 'intent_database_' + split_name + '.pkl'
        with open(os.path.join(args.database_path, database_name), 'wb') as fid:
            pickle.dump(db, fid)

    print("Finished collecting database!")


def add_ped_case(db, video_name, ped_name, nlp_vid_uid_pairs):
    if video_name not in db:
        db[video_name] = {}

    db[video_name][ped_name] = {  # ped_name is 'track_id' in cv-annotation
        'frames': None,  # [] list of frame_idx of the target pedestrian appear
        'cv_annotations': {
            'track_id': ped_name,
            'bbox': []  # [] list of bboxes, each bbox is [xtl, ytl, xbr, ybr]
        },
        'nlp_annotations': {
            # [vid_uid_pair: {'intent': [], 'description': [], 'key_frame': []}]
        }
    }
    for vid_uid in nlp_vid_uid_pairs:
        db[video_name][ped_name]['nlp_annotations'][vid_uid] = {
            'intent': [],
            'description': [],
            'key_frame': []
            # 0: not key frame (expanded from key frames with NLP annotations)
            # 1: key frame (labeled by NLP annotations)
        }



def init_db(video_list, db_log, args):
    db = {}
#     data_split = 'train' # 'train', 'val', 'test'
    dataroot = args.dataset_root_path
    # key_frame_folder = 'cognitive_annotation_key_frame'
    if args.dataset == 'PSI2.0':
        extended_folder = 'PSI2.0_TrainVal/annotations/cognitive_annotation_extended'
    elif args.dataset == 'PSI1.0':
        extended_folder = 'PSI1.0/annotations/cognitive_annotation_extended'

    for video_name in sorted(video_list):
        try:
            with open(os.path.join(dataroot, extended_folder, video_name, 'pedestrian_intent.json'), 'r') as f:
                annotation = json.load(f)
        except:
            with open(db_log, 'a') as f:
                f.write(f"Error loading {video_name} pedestrian intent annotation json \n")
            continue
        db[video_name] = {}
        for ped in annotation['pedestrians'].keys():
            cog_annotation = annotation['pedestrians'][ped]['cognitive_annotations']
            nlp_vid_uid_pairs = cog_annotation.keys()
            add_ped_case(db, video_name, ped, nlp_vid_uid_pairs)
    return db


def split_frame_lists(frame_list, bbox_list, threshold=60):
    # For a sequence of an observed pedestrian, split into slices based on missingframes
    frame_res = []
    bbox_res = []
    inds_res = []

    inds_split = [0]
    frame_split = [frame_list[0]]  # frame list
    bbox_split = [bbox_list[0]]  # bbox list
    for i in range(1, len(frame_list)):
        if frame_list[i] - frame_list[i - 1] == 1: # consistent
            inds_split.append(i)
            frame_split.append(frame_list[i])
            bbox_split.append(bbox_list[i])
        else:  # # next position frame is missing observed
            if len(frame_split) > threshold:  # only take the slices longer than threshold=max_track_length=60
                inds_res.append(inds_split)
                frame_res.append(frame_split)
                bbox_res.append(bbox_split)
                inds_split = []
                frame_split = []
                bbox_split = []
            else:  # ignore splits that are too short
                inds_split = []
                frame_split = []
                bbox_split = []
    # break loop when i reaches the end of list
    if len(frame_split) > threshold:  # reach the end
        inds_res.append(inds_split)
        frame_res.append(frame_split)
        bbox_res.append(bbox_split)

    return frame_res, bbox_res, inds_res


def get_intent_des(db, vname, pid, split_inds, cog_annt):
    # split_inds: the list of indices of the intent_annotations for the current split of pid in vname
    for vid_uid in cog_annt.keys():
        intent_list = cog_annt[vid_uid]['intent']
        description_list = cog_annt[vid_uid]['description']
        key_frame_list = cog_annt[vid_uid]['key_frame']

        nlp_vid_uid = vid_uid
        db[vname][pid]['nlp_annotations'][nlp_vid_uid]['intent'] = [intent_list[i] for i in split_inds]
        db[vname][pid]['nlp_annotations'][nlp_vid_uid]['description'] = [description_list[i] for i in split_inds]
        db[vname][pid]['nlp_annotations'][nlp_vid_uid]['key_frame'] = [key_frame_list[i] for i in split_inds]


def update_db_annotations(db, db_log, args):
    dataroot = args.dataset_root_path
    # key_frame_folder = 'cognitive_annotation_key_frame'
    if args.dataset == 'PSI2.0':
        extended_folder = 'PSI2.0_TrainVal/annotations/cognitive_annotation_extended'
    elif args.dataset == 'PSI1.0':
        extended_folder = 'PSI1.0/annotations/cognitive_annotation_extended'

    video_list = sorted(db.keys())
    for video_name in video_list:
        ped_list = list(db[video_name].keys())
        tracks = list(db[video_name].keys())
        try:
            with open(os.path.join(dataroot, extended_folder, video_name, 'pedestrian_intent.json'), 'r') as f:
                annotation = json.load(f)
        except:
            with open(db_log, 'a') as f:
                f.write(f"Error loading {video_name} pedestrian intent annotation json \n")
            continue

        for pedId in ped_list:
            observed_frames = annotation['pedestrians'][pedId]['observed_frames']
            observed_bboxes = annotation['pedestrians'][pedId]['cv_annotations']['bboxes']
            cog_annotation = annotation['pedestrians'][pedId]['cognitive_annotations']
            if len(observed_frames) == observed_frames[-1] - observed_frames[0] + 1: # no missing frames
                threshold = args.max_track_size # 16 for intent/driving decision; 60 for trajectory
                if len(observed_frames) > threshold:
                    cv_frame_list = observed_frames
                    cv_frame_box = observed_bboxes
                    db[video_name][pedId]['frames'] = cv_frame_list
                    db[video_name][pedId]['cv_annotations']['bbox'] = cv_frame_box
                    get_intent_des(db, video_name, pedId, [*range(len(observed_frames))], cog_annotation)
                else: # too few frames observed
                    # print("Single ped occurs too short.", video_name, pedId, len(observed_frames))
                    with open(db_log, 'a') as f:
                        f.write(f"Single ped occurs too short. {video_name}, {pedId}, {len(observed_frames)} \n")
                    del db[video_name][pedId]
            else: # missing frames exist
                with open(db_log, 'a') as f:
                    f.write(f"missing frames bbox noticed! , {video_name}, {pedId}, {len(observed_frames)}, frames observed from , {observed_frames[-1] - observed_frames[0] + 1} \n")
                threshold = args.max_track_size  # 60
                cv_frame_list, cv_frame_box, cv_split_inds = split_frame_lists(observed_frames, observed_bboxes, threshold)
                if len(cv_split_inds) == 0:
                    with open(db_log, 'a') as f:
                        f.write(f"{video_name}, {pedId}, After removing missing frames, no split left! \n")

                    del db[video_name][pedId]
                elif len(cv_split_inds) == 1:
                    db[video_name][pedId]['frames'] = cv_frame_list[0]
                    db[video_name][pedId]['cv_annotations']['bbox'] = cv_frame_box[0]
                    get_intent_des(db, video_name, pedId, cv_split_inds[0], cog_annotation)
                else:
                    # multiple splits left after removing missing box frames
                    with open(db_log, 'a') as f:
                        f.write(f"{len(cv_frame_list)} splits: , {[len(s) for s in cv_frame_list]} \n")
                    nlp_vid_uid_pairs = db[video_name][pedId]['nlp_annotations'].keys()
                    for i in range(len(cv_frame_list)):
                        ped_splitId = pedId + '-' + str(i)
                        add_ped_case(db, video_name, ped_splitId, nlp_vid_uid_pairs)
                        db[video_name][ped_splitId]['frames'] = cv_frame_list[i]
                        db[video_name][ped_splitId]['cv_annotations']['bbox'] = cv_frame_box[i]
                        get_intent_des(db, video_name, ped_splitId, cv_split_inds[i], cog_annotation)
                        if len(db[video_name][ped_splitId]['nlp_annotations'][list(db[video_name][ped_splitId]['nlp_annotations'].keys())[0]]['intent']) == 0:
                            raise Exception("ERROR!")
                    del db[video_name][pedId] # no pedestrian list left, remove this video
            tracks.remove(pedId)
        if len(db[video_name].keys()) < 1: # has no valid ped sequence! Remove this video!")
            with open(db_log, 'a') as f:
                f.write(f"!!!!! Video, {video_name}, has no valid ped sequence! Remove this video! \n")
            del db[video_name]
        if len(tracks) > 0:
            with open(db_log, 'a') as f:
                f.write(f"{video_name} missing pedestrians annotations: {tracks}  \n")


# def cut_sequence(db, db_log, args):
#     # only wanna use some of the sequence, thus cut edges
#     for vname in sorted(db.keys()):
#         for pid in sorted(db[vname].keys()):
#             frames = db[vname][pid]['frames']
#             first_cog_idx = len(frames) + 1
#             last_cog_idx = -1
#             for uv in db[vname][pid]['nlp_annotations'].keys():
#                 key_frame_list = db[vname][pid]['nlp_annotations'][uv]['key_frame']
#                 for i in range(len(key_frame_list)):
#                     if key_frame_list[i] == 1:
#                         first_cog_idx = min(first_cog_idx, i)
#                         break
#                 for j in range(len(key_frame_list)-1, -1, -1):
#                     if key_frame_list[j] == 1:
#                         last_cog_idx = max(last_cog_idx, j)
#                         break
#
#             if first_cog_idx > len(frames) or last_cog_idx < 0:
#                 print("ERROR! NO key frames found in ", vname, pid)
#             else:
#                 print("First and last annotated key frames are: ", frames[first_cog_idx], frames[last_cog_idx])
#                 print('In total frames # = ', last_cog_idx - first_cog_idx, ' out of ', len(frames))
#
#             if last_cog_idx - first_cog_idx < args.max_track_size:
#                 print(vname, pid, " too few frames left after cutting sequence.")
#                 del db[vname][pid]
#             else:
#                 db[vname][pid]['frames'] = db[vname][pid]['frames'][first_cog_idx: last_cog_idx+1]
#                 db[vname][pid]['cv_annotations']['bbox'] = db[vname][pid]['cv_annotations']['bbox'][first_cog_idx: last_cog_idx + 1]
#                 db[vname][pid]['frames'] = db[vname][pid]['frames'][first_cog_idx: last_cog_idx + 1]
#                 for uv in db[vname][pid]['nlp_annotations'].keys():
#                     db[vname][pid]['nlp_annotations'][uv]['intent'] = db[vname][pid]['nlp_annotations'][uv]['intent'][first_cog_idx: last_cog_idx + 1]
#                     db[vname][pid]['nlp_annotations'][uv]['description'] = db[vname][pid]['nlp_annotations'][uv]['description'][
#                                                           first_cog_idx: last_cog_idx + 1]
#                     db[vname][pid]['nlp_annotations'][uv]['key_frame'] = db[vname][pid]['nlp_annotations'][uv]['key_frame'][
#                                                                       first_cog_idx: last_cog_idx + 1]
#         if len(db[vname].keys()) < 1:
#             print(vname, "After cutting sequence edges, not enough frames left! Delete!")
#             del db[vname]

