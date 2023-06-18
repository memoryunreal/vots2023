import os
import glob
import struct

sequences_dir = '/home/dataset/vots2023/sequences_ori/'
sequences_gt_dir = '/home/dataset/vots2023/sequences'
dir_list = os.listdir(sequences_dir)
seqlist = [item for item in dir_list if os.path.isdir(os.path.join(sequences_dir, item))]
seqlist.sort()

annotation_gt = '/home/paper/vots2023/workspace/results/vot_bin/baseline/'
annotation_gt_list = os.listdir(annotation_gt)
annotation_gt_list.sort()


def copy_color():
    for seq in seqlist:
        seq_color = os.path.join(sequences_dir, seq, 'color')
        softlink_color = os.path.join(sequences_gt_dir, seq, 'color')
        seq_att = os.path.join(sequences_dir, seq, 'sequence')
        softlink_att = os.path.join(sequences_gt_dir, seq, 'sequence')
        softlink_seq = os.path.join(sequences_gt_dir, seq)
        os.makedirs(softlink_seq, exist_ok=True)
        os.symlink(seq_color, softlink_color)
        os.symlink(seq_att, softlink_att)
    
def gt_convert():
    for seq in annotation_gt_list:
        bin_files = glob.glob(os.path.join(annotation_gt, seq) + '/*.bin')
        bin_files.sort()
        if len(bin_files) == 1:
            continue

copy_color()


