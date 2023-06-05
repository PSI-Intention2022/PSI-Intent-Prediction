import argparse


def get_opts():
    parser = argparse.ArgumentParser(description='PyTorch implementation of the PSI2.0')
    # about data
    parser.add_argument('--dataset', type=str, default='PSI2.0',
                        help='task name: [PSI1.0 | PSI2.0]')
    parser.add_argument('--task_name', type=str, default='ped_intent',
                        help='task name: [ped_intent | ped_traj | driving_decision]')
    parser.add_argument('--video_splits', type=str, default='./splits/PSI200_split.json',
                        help='video splits, [PSI100_split | PSI200_split | PSI200_split_paper]')
    parser.add_argument('--dataset_root_path', type=str, default='path to dataset rootpath',
                        help='Path of the dataset, e.g. frames/video_0001/000.jpg')
    parser.add_argument('--database_path', type=str, default='./database',
                        help='Path of the database created based on the cv_annotations and nlp_annotations')
    parser.add_argument('--database_file', type=str, default='intent_database_train.pkl',
                        help='Filename of the database created based on the cv_annotations and nlp_annotations')
    parser.add_argument('--fps', type=int, default=30,
                        help=' fps of original video, PSI and PEI == 30.')
    parser.add_argument('--seq_overlap_rate', type=float, default=0.9, # 1 means every stride is 1 frame
                        help='Train/Val rate of the overlap frames of slideing windown, (1-rate)*seq_length is the step size')
    parser.add_argument('--test_seq_overlap_rate', type=float, default=1, # 1 means every stride is 1 frame
                        help='Test overlap rate of the overlap frames of slideing windown, (1-rate)*seq_length is the step size')
    parser.add_argument('--intent_num', type=int, default=2,
                        help='Type of intention categories. [2: {cross/not-cross} | 3 {not-sure}]')
    parser.add_argument('--intent_type', type=str, default='mean',
                        help='Type of intention labels, out of 24 annotators. [major | mean | separate | soft_vote];'
                             'only when separate, the nlp reasoning can help, otherwise may take weighted mean of the nlp embeddings')
    parser.add_argument('--observe_length', type=float, default=15,
                        help='Sequence length of one observed clips')
    parser.add_argument('--predict_length', type=float, default=45,
                        help='Sequence length of predicted trajectory/intention')
    parser.add_argument('--max_track_size', type=float, default=60,
                        help='Sequence length of observed + predicted trajectory/intention')
    parser.add_argument('--crop_mode', type=str, default='enlarge',
                        help='Cropping mode of cropping the pedestrian surrounding area')
    parser.add_argument('--balance_data', type=bool, default=False,
                        help='Balance data sampler with randomly class-wise weighted')
    parser.add_argument('--normalize_bbox', type=str, default=None,
                        help='If normalize bbox. [L2 | subtract_first_frame | divide_image_size]')
    parser.add_argument('--image_shape', type=tuple, default=(1280, 720),
                        help='Image shape: PSI(1280, 720).')
    parser.add_argument('--load_image', type=bool, default=False,
                        help='Do not load image to backbone if not necessary')

    # about models
    parser.add_argument('--backbone', type=str, default='',
                        help='Backbone type [resnet50 | vgg16 | faster_rcnn]')
    parser.add_argument('--freeze_backbone', type=bool, default=False,
                        help='[True | False]')
    parser.add_argument('--model_name', type=str, default='lstm',
                        help='model name, [lstm, lstmed]')
    parser.add_argument('--intent_model', type=bool, default=True,
                        help='[True | False]')
    parser.add_argument('--traj_model', type=bool, default=False,
                        help='[True | False]')
    parser.add_argument('--model_configs', type=dict, default={},
                        help='framework information')

    # about training
    parser.add_argument('--checkpoint_path', type=str, default='./ckpts',
                        help='Path of the stored checkpoints')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Total number of training epochs')
    parser.add_argument('--batch_size', type=float, default=128,
                        help='Batch size of dataloader')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='General learning rate, default as 1e-3')
    parser.add_argument('--resume', type=str, default='',
                        help='ckpt path+filename to be resumed.')
    parser.add_argument('--loss_weights', type=dict, default={},
                        help='weights of loss terms, {loss_intent, loss_traj}')
    parser.add_argument('--intent_loss', type=list, default=['bce'],
                        help='loss for intent output. [bce | mse | cross_entropy]')
    parser.add_argument('--intent_disagreement', type=float, default=-1.0,
                        help='weather use disagreement to reweight intent loss.threshold to filter training data.'
                             'consensus > 0.5 are selected and reweigh loss; -1.0 means not use; 0.0, means all are used.')
    parser.add_argument('--ignore_uncertain', type=bool, default=False,
                        help='ignore uncertain training samples, based on intent_disagreement')
    parser.add_argument('--intent_positive_weight', type=float, default=1.0,
                        help='weight for intent bce loss: e.g., 0.5 ~= n_neg_class_samples(5118)/n_pos_class_samples(11285)')
    parser.add_argument('--traj_loss', type=list, default=['mse'],
                        help='loss for intent output. [bce | mse | cross_entropy]')

    # other parameteres
    parser.add_argument('--val_freq', type=int, default=10,
                        help='frequency of validate')
    parser.add_argument('--test_freq', type=int, default=10,
                        help='frequency of test')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='frequency of print')

    return parser.parse_args()
