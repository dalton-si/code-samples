import configparser
import os
import re
import sys
import xml.etree.cElementTree as ET
from glob import glob

import pandas as pd

sys.path.append('../../..')
sys.path.append("/mnt/CEDAR-Lab-data/Projects/PHdeid/deep_learning_deid")

from sklearn.model_selection import train_test_split

from src.data.bio_tagger import get_bio_labeled_tokens
from src.data.uofm.uofm_preprocess import clean_text_and_labels
from src.data.i2b2.helper import extract_phi_tags, get_text


def main():
    """
    
    prepare_uofm: 
        - Converts Clinical (.txt) and their annotations (.ann) into XML, then splits the XML files in a Train directory and Test directory
        - Loads the XML and converts into BIO format for use in model training and inference

    
    Command Line Args:
        prepare_uofm_config: .ini configuration file directory information and preprocessing parameters


    Returns:
        - A directory of XML training files
        - A directory of XML test files
        - A directory of BIO training files
        - A directory of BIO test files 

    """
    TXT_DIR, ANN_DIR, XML_TRAIN_DIR, XML_TEST_DIR, BIO_DIR, PREPROCESS, TEST_PARAM, TEST_DEV_SPLIT, DEMO_TRAIN_FP, \
    DEMO_TEST_FP, DEMO_TEST_SUBGROUP1_FP, DEMO_TEST_SUBGROUP2_FP = load_config()
    print(PREPROCESS, TEST_PARAM, TEST_DEV_SPLIT)
    write_xml_files(TXT_DIR, ANN_DIR, XML_TRAIN_DIR, XML_TEST_DIR, test_param=TEST_PARAM, demo_train_fp=DEMO_TRAIN_FP,
                    demo_test_fp=DEMO_TEST_FP)

    train_files, _test_files = get_xml_files(XML_TRAIN_DIR), get_xml_files(XML_TEST_DIR)

    if TEST_DEV_SPLIT > 0:
        test_files, dev_files = train_test_split(_test_files, test_size=TEST_DEV_SPLIT, random_state=1)
        _, dev_labels = xml2bio(dev_files, output_file=BIO_DIR + 'dev.txt', preprocess=PREPROCESS)
    else:
        test_files = _test_files
        dev_labels = []

    if DEMO_TEST_SUBGROUP1_FP != 'None':
        _, test_group1_labels = test_subgroup_xml2bio(test_files, BIO_DIR, DEMO_TEST_SUBGROUP1_FP, 'group1')
    if DEMO_TEST_SUBGROUP2_FP != 'None':
        _, test_group12_labels = test_subgroup_xml2bio(test_files, BIO_DIR, DEMO_TEST_SUBGROUP2_FP, 'group2')

    _, train_labels = xml2bio(train_files, output_file=BIO_DIR + 'train.txt', preprocess=PREPROCESS)
    _, test_labels = xml2bio(test_files, output_file=BIO_DIR + 'test.txt', preprocess=PREPROCESS)

    label_set_2_file(train_labels, dev_labels, test_labels, output_file=BIO_DIR + 'labels.txt')


def load_config():
    """

    load_config:
        loads a .ini configuration file. If none provided, generates a new .ini file with a default configuration
    

    Command Line Args:
        prepare_uofm_config: .ini configuration file directory information and preprocessing parameters


    Returns:
        See print_config_explanation() for readout of prepare_uofm_config parameters

    """
    config = configparser.ConfigParser()

    def write_file():
        config.write(open('prepare_uofm_config.ini', 'w'))

    if not os.path.exists('prepare_uofm_config.ini'):
        print("New prepare_uofm_config.ini file generated; please complete input fields")
        config['PARAMS'] = {
            'TXT_DIR': '/home/[user]/source_data/',  # Directory to raw
            'ANN_DIR': '/home/[user]/source_data/',
            'XML_TRAIN_DIR': '/home/[user]/source_data/',
            'XML_TEST_DIR': '/home/[user]/source_data/',
            'BIO_DIR': '/home/[user]/source_data/',
            'PREPROCESS': True,
            'TEST_PARAM': float(0.4),
            'TEST_DEV_SPLIT': float(0.5),
            'DEMO_TRAIN_FP': '',
            'DEMO_TEST_FP': '',
            'DEMO_TEST_SUBGROUP1_FP': 'None',
            'DEMO_TEST_SUBGROUP2_FP': 'None'

        }
        write_file()
        print_config_explanation()
        sys.exit()
    else:
        config.read('prepare_uofm_config.ini')
        TXT_DIR = config['PARAMS']['TXT_DIR']
        ANN_DIR = config['PARAMS']['ANN_DIR']
        XML_TRAIN_DIR = config['PARAMS']['XML_TRAIN_DIR']
        XML_TEST_DIR = config['PARAMS']['XML_TEST_DIR']
        BIO_DIR = config['PARAMS']['BIO_DIR']
        PREPROCESS = config.getboolean('PARAMS', 'PREPROCESS')
        TEST_PARAM = config.getfloat('PARAMS', 'TEST_PARAM')
        TEST_DEV_SPLIT = config.getfloat('PARAMS', 'TEST_DEV_SPLIT')
        DEMO_TRAIN_FP = config['PARAMS']['DEMO_TRAIN_FP']
        DEMO_TEST_FP = config['PARAMS']['DEMO_TEST_FP']
        DEMO_TEST_SUBGROUP1_FP = config['PARAMS']['DEMO_TEST_SUBGROUP1_FP']
        DEMO_TEST_SUBGROUP2_FP = config['PARAMS']['DEMO_TEST_SUBGROUP2_FP']

        return TXT_DIR, ANN_DIR, XML_TRAIN_DIR, XML_TEST_DIR, BIO_DIR, PREPROCESS, TEST_PARAM, TEST_DEV_SPLIT, \
               DEMO_TRAIN_FP, DEMO_TEST_FP, DEMO_TEST_SUBGROUP1_FP, DEMO_TEST_SUBGROUP2_FP


def print_config_explanation():
    print("""
        PARAMETER Explanations
            TXT_DIR: Directory path to raw note .txt files 
            ANN_DIR: Directory path to raw note annotation .ann files
            XML_TRAIN_DIR: Target path to output training xml files
            XML_TEST_DIR: Target path to output test xml files
            BIO_DIR: Target path to output bio-labled files
            TEST_PARAM: OPTIONAL (FLOAT) - proportion to split to test data when train/test was NOT done with MRN sampling
            TEST_DEV_SPLIT: (FLOAT) - proportion of test data to split into dev

            # Used if notes have been resampled via sample_by_MRN.py 
            DEMO_TRAIN_FP: OPTIONAL: File path to MRNs used in training (i.e. output from sample_by_MRN.py)
            DEMO_TEST_FP: OPTIONAL: File path to MRNs used in test (i.e. output from sample_by_MRN.py) 


            # Additional file path to test subgroups
            DEMO_TEST_SUBGROUP1_FP: OPTIONAL: File path to subgroup of test data (i.e. output from sample_by_MRN.py)
            DEMO_TEST_SUBGROUP2_FP: OPTIONAL: File path to subgroup of test data (i.e. output from sample_by_MRN.py)

    """)
    return None


def test_subgroup_xml2bio(test_files, bio_dir, demo_test_subgroup_fp, label, preprocess=True):
    """
    
    test_subgroup_xml2bio: create subset of test cases based on pre-determined list of patients

    Args:
        test_files: list of xml file paths
        bio_dir: Output directory
        demo_test_subgroup_fp: file path to list of patient Identifiers
        label: label name for test group
        preprocess: 

    Returns:
        tokens: list of labeled tokens
        labels: set of unique labels found in annotated text files

    """

    # Read in predetermined list of patients
    subgroup_test_MRN_list = pd.read_csv(demo_test_subgroup_fp)['PAT_MRN_ID'].to_list()
    
    # Filter test files to those with patients in the target list
    subtest_files = [fp for fp in test_files if int(fp.split('/')[-1].split('_')[0]) in subgroup_test_MRN_list]
    
    # Return the labeled tokens in those test files with the list of unqiue labels
    tokens, labels = xml2bio(subtest_files, output_file=bio_dir + f'test_{label}.txt', preprocess=preprocess)
    
    return tokens, labels


def xml2bio(xml_files, output_file, preprocess=True):
    """ 
    
    xml2bio: loads xml files, preprocess contents and output to a BIO-formatted file

    Args:
        test_files: list of xml file paths
        output_file: BIO Output filepath

    Returns:
        labeled_tokens_by_file: list of labeled tokens
        labels: set of unique labels found in annotated text files

    """
    print("Converting {} XML files to BIO formatted file {}".format(len(xml_files), output_file))
    labeled_tokens_by_file = []
    label_set = set()

    for file_path in xml_files:
        if preprocess:
            text, phi_tags = clean_text_and_labels(file_path)
        else:
            text = get_text(file_path)
            phi_tags = extract_phi_tags(file_path)

        labeled_tokens = get_bio_labeled_tokens(text, phi_tags)
        labeled_tokens_by_file.append(labeled_tokens)
        label_set = label_set.union([x[2] for x in labeled_tokens])

    print("Saving BIO file to {}".format(output_file))
    bio_labels_2_file(labeled_tokens_by_file, output_file)
    return labeled_tokens_by_file, label_set


def bio_labels_2_file(labeled_tokens_by_file, output_file, max_seq_len=200):
    with open(output_file, 'w') as out:
        for labeled_tokens in labeled_tokens_by_file:
            out.write("-DOCSTART- -X- -X- O\n\n")
            sentence_len = 0
            for i, (token, pos, label) in enumerate(labeled_tokens):
                out.write("{} {} {}\n".format(str(token), pos, label))
                sentence_len += 1
                if is_end_of_sentence(labeled_tokens, i) or sentence_len >= max_seq_len:
                    out.write("\n")
                    sentence_len = 0
            out.write("\n")


def is_end_of_sentence(tags, i):
    honorifics = ['mr', 'ms', 'mrs', 'miss', 'dr']
    if i > 1:
        previous_token = tags[i - 1][0]
        token, pos, label = tags[i]
        if token == '.' and previous_token.lower() not in honorifics:
            return True
    return False


def get_xml_files(dir_path):
    xml_files = []
    for file_path in glob(dir_path + "*.xml"):
        xml_files.append(file_path)
    return xml_files


def get_doc_ids(dir_path: str):
    ''' Strips and returns set of file ids from text document file names '''
    doc_ids = []
    for file_path in glob(dir_path + "*.txt"):
        doc_id = re.search(r"(?<=\/)[\d_]+(?=\.txt)", file_path).group()

        if len(doc_id.split('_')) == 1:  # MRN not in proper format
            continue

        doc_ids.append(doc_id)

    return doc_ids


def write_xml_files(txt_path: str, ann_path: str, train_path: str, test_path: str, test_param=0.4, demo_train_fp='',
                    demo_test_fp=''):

    """

        write_xml_files: concatenates clinical (.txt) data with its PHI annotation labels (.ann) to XML file; splits output into Train or Test directories


    Args:
        txt_path: Path to source .txt clinical text files
        ann_path: Path to clinical text annotations
        train_path: Output path for training XML
        test_path: Output path for testing XML
        test_param: float, proportion of test set
        demo_train_fp: filepath for demographic splits in training data
        demo_test_fp: filepath for demographic splits in test data

    """

    # parse filenames for ids.
    doc_ids = get_doc_ids(txt_path)

    # if train test split was done during MRN demographic preprocessing (Filepaths to train demo file and test demo file available)
    if demo_test_fp != '':
        test_demo_df = pd.read_csv(demo_test_fp)
        test_ids = [id for id in doc_ids if int(id.split('_')[0]) in test_demo_df['PAT_MRN_ID'].to_list()]

    else:
        _, test_ids = train_test_split(doc_ids, test_size=test_param, random_state=1)

    if demo_train_fp != '':
        train_demo_df = pd.read_csv(demo_train_fp)
        doc_ids = [id for id in doc_ids if int(id.split('_')[0]) in train_demo_df['PAT_MRN_ID'].to_list()] + test_ids


    # for each doc_id
    for doc_id in doc_ids:

        # load its text file
        txt_file_path = "{}{}.txt".format(txt_path, doc_id)
        txt_f = open(txt_file_path, 'r')
        txt_file = txt_f.readlines()

        # and its annotation file.
        ann_file_path = "{}{}.ann".format(ann_path, doc_id)
        ann_f = open(ann_file_path, 'r')
        ann_file = ann_f.readlines()

        # structure xml
        root = ET.Element("uofm_deId")

        # with source txt
        txt = ET.SubElement(root, "TEXT").text = txt_file[0]

        # and each annotation tag
        ann = ET.SubElement(root, "TAGS")
        for line in ann_file:

            # NOTE: ann file use mixed delimiters, e.g. T4\tDATE 1024 1033\t8/29/2019
            if line[0] != '#':
                id, TYPE_start_end, text_comment = line.split('\t')
                TYPE, start, end = TYPE_start_end.split(' ')
                text, comment = text_comment, ""  # TODO: do not concatenate comments to PHI text
                ET.SubElement(ann, "PHI", id=id, start=start, end=end, text=text, TYPE=TYPE, comment=comment)

        # then output to xml.
        tree = ET.ElementTree(root)
        xml_file_str = "{}.xml".format(doc_id)
        if doc_id in test_ids:
            tree.write(test_path + xml_file_str)
        else:
            tree.write(train_path + xml_file_str)


def label_set_2_file(train_labels, dev_labels, test_labels, output_file):
    """ Outputs a text file listing the unique set of BIO labels found among train, dev and test data """
    label_set = set().union(train_labels, dev_labels, test_labels)
    print("Saving BIO label set to {}".format(output_file))
    with open(output_file, 'w') as out:
        for label in label_set:
            out.write("{}\n".format(label))


if __name__ == '__main__':
    main()
