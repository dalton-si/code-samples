from collections import OrderedDict
import flwr as fl
import os
import numpy as np
import multiprocessing as mp
from collections import Counter

from flwr.server.strategy import FedAvg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd

from torch.utils.data import DataLoader, random_split
from datasets import load_metric

from transformers import LongformerTokenizer, LongformerForTokenClassification, DataCollatorForTokenClassification, BertTokenizer, LongformerTokenizerFast, TrainingArguments, Trainer, set_seed, pipeline
from torch.optim import AdamW
from src.data.uofm.prepare_uofm import get_xml_files
from src.models.transformers_ner.utils_ner import read_examples_from_file, convert_examples_to_features, get_labels, InputExample, get_labels
from src.data.uofm.prepare_uofm import xml2bio
from src.models.transformers_ner.ner_dataset import NERDataset
from src.models.transformers_ner.model_predict import run_pipeline_on_bio
from src.data.bio_tagger import tag_text_from_pipeline_entities
from src.models.post_model_regexes.tagger import run_tagger_on_labeled_tokens

from tqdm import tqdm
from evaluate import load as load_metric

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def get_params(model):
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


# Function to set the weights of a model
def set_params(model, weights):
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def batch_data(config, NERDataset, shuffle=True):
    """ Creates batched Dataloader from NERDataset """
    batch_size = config.getint('model', 'batch_size')

    _, collator, _ = create_model_components(config)


    data_loader = DataLoader(
        dataset=NERDataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=shuffle
    )

    return data_loader


def create_model_components(config, unique_tags=None):
    """
    Create components for Token Classifier
        NOTE: Supports Longformer Variants

    Parameters
    __________
    config: Config Object
        User set configurations in .ini config file

        REQUIRED FIELDS
        _______________
        model_name: model string to load from huggingface
        is_longformer: indicates whether model_name is a longformer variant of token classifier

    unique_tags
        Set of class labels for token classifier
            Required for instantiating a model; returns None Otherwise


    Returns
    _______
    tokenizer
        tokenizer based on a valid model_name

    collator
        collator derived from instantiated tokenizer

    model
        pre-trained instantiation of a valid model
            Requires unique_tags
    """
    tokenizer, collator, model = None, None, None

    model_name = config['model']['model_name_or_path']

    seed = config.getint('model', 'seed')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    set_seed(seed)

    if unique_tags:
        label2id = {l: i for i, l in enumerate(unique_tags)}
        id2label = {value: key for key, value in label2id.items()}

    is_longformer = config.getboolean('model', 'is_longformer')


    if is_longformer:
        tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
        collator = DataCollatorForTokenClassification(tokenizer)
        if unique_tags:
            model = LongformerForTokenClassification.from_pretrained(
                model_name,
                id2label=id2label,
                label2id=label2id
            )
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        collator = DataCollatorForTokenClassification(tokenizer)
        if unique_tags:
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                id2label=id2label,
                label2id=label2id
            )

    def model_init():
        return model

    return tokenizer, collator, model_init()




def train(cid, net, epochs, return_dict, config, trainloader, toy, verbose=True):

    """
    train
        train transformer model

    Args:
        cid: unique client identifier
        net: the preploaded model object
        epochs: the number of epochs
        return_dict: dictionary of training result variables
        config: pre-loaded user-config .ini file
        toy: boolean for test run

    Returns:
        return_dict: Dict for function return values
            keys
                totol_loss: calculated loss from test set inference
                accuracy: calculated micro accuracy across all example words
                data_size: count of examples
                results: list of tuple results with each as (word, true label, predicted label)
    """

    print(f"Starting training on {DEVICE}...")
    net.to(DEVICE)
    learning_rate = config.getfloat('model','learning_rate')
    data_loader = batch_data(config, trainloader)
    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = AdamW(net.parameters(), lr=learning_rate)

    net.train()
    for epoch in range(epochs):
        print(f"Client {cid}: Training on Epoch {epoch}")
        correct, total, epoch_loss = 0, 0, 0.0

        for batch in tqdm(data_loader):

            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            print(batch['labels'])

            print(batch['labels'].shape)

            #print(len(predicted_labels))
            sys.exit()
            optimizer.zero_grad()
            outputs = net(inputs)#, labels=labels)
            predicted_labels = torch.argmax(outputs.logits, dim=-1).to(DEVICE)


            sys.exit()

            loss = criterion(predicted_labels.to(dtype=torch.double), labels.to(dtype=torch.double))
            loss.backward
            optimizer.step()
            epoch_loss += loss
            total += labels.size(1) * labels.size(0)
            correct += (torch.flatten(predicted_labels) == torch.flatten(labels)).sum().item()
            print(f"    Batch running epoch loss {epoch_loss} and running epoch accuracy {correct / total}")
            if toy:
                break
        epoch_loss /= len(trainloader)
        print(correct)
        print(total)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    loss /= len(trainloader)
    accuracy = correct / total
    return_dict["parameters"] = get_params(net)
    return_dict["data_size"] = len(trainloader)
    return_dict["accuracy"] = float(accuracy)

    net.to("cpu")  # move model back to CPU




def test(config, examples, params, unique_tags, return_dict, toy=False):
    """
    test
        Infer class predictions per word in each example sequence
        return results and performance metrics

    Args:
        config: parsed user_config form .ini file
        examples: list of InputExamples
        params: list of numpy weight arrays (loaded from current model)
        unique_tags: list of unique BIO labels
        return_dict: Dict for function return values
        toy: boolean for test run

    Returns:
        return_dict: Dict for function return values
            keys
                totol_loss: calculated loss from test set inference
                accuracy: calculated micro accuracy across all example words
                data_size: count of examples
                results: list of tuple results with each as (word, true label, predicted label)
    """
    toy_idx = 10 if toy else None

    max_seq_len = config.getint('model','max_seq_len')

    total_loss, correct, total = 0.0, 0.0, 0.0

    tokenizer, _, model = create_model_components(config, unique_tags)
    set_params(model, params)

    all_words = []
    all_true_labels = []
    all_predictions = []

    # Load words and their labels
    for example in examples[:toy_idx]:
        all_words.extend(example.words)
        all_true_labels.extend(example.labels)


    # Iterate through words-labels per sequence of max_seq_len
    for i in range(0, len(all_words), max_seq_len)[:toy_idx]:
        example_seq = all_words[i:i + max_seq_len]
        example_labels = all_true_labels[i:i + max_seq_len]


        # Prepare word sequence for inference
        text = " ".join(example_seq)
        inputs = tokenizer(text, return_tensors="pt")
        print(inputs)
        import sys
        sys.exit()
        tokens = tokenizer.tokenize(text) # Ġ not inserted for first token in longformer

        # Inference Step
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Post-process inference predictions
        pred_logits = aggregate_subtoken_logits(tokens, logits, len(example_seq))
        label2id = model.config.label2id
        class_labels = [label2id[i] for i in example_labels]
        class_labels_tensor = torch.tensor(class_labels)
        one_hot_labels = torch.nn.functional.one_hot(class_labels_tensor, len(unique_tags)).to(torch.float32).unsqueeze(0)

        # Calculate Loss and Performance Metric Inputs
        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(pred_logits, one_hot_labels).item()
        total_loss += loss
        total += len(example_seq)
        word_probabilities = torch.softmax(pred_logits, dim=-1)
        word_predictions = torch.argmax(word_probabilities, dim=-1).flatten().tolist()
        correct += (sum(x == y for x, y in zip(word_predictions, class_labels)))

        # Add Predictions to result set
        id2label = model.config.id2label
        example_preds = [id2label[i] for i in word_predictions]
        all_predictions.extend(example_preds)

    assert len(all_words) == len(all_true_labels) == len(all_predictions), f"{len(all_words)},{len(all_true_labels)},{len(all_predictions)}"
    results = list(zip(all_words, all_true_labels, all_predictions))
    accuracy = correct / total
    return_dict["loss"] = total_loss
    return_dict["accuracy"] = accuracy
    return_dict["data_size"] = len(examples)
    return_dict["results"] = results




def aggregate_subtoken_logits(tokens, logits, word_sequence_length, type='longformer'):
    """
    aggregate_subtoken_logits
        Aggregate subtoken logits per word in an labeled word sequence

    Args:
        tokens: subtoken output from HuggingFace Tokenizer
        logits: raw transformer model inference logit outputs of all subtokens
        type: varaible to describe the branch of tokenizer used (e.g. Longformer uses Ġ char to mark a new word)
        word_sequence_length: length of source word sequence
        strategy: the type of logit aggregation
            defaults to average

    Returns:
        final_logits: a logit of shape (1,word_sequence_length) where subtoken logits have been aggregated
    """

    # Unique token character added from tokenizer type, e.g. longerformer tokenizer adds Ġ to subtokens that start a new word
    if type == 'longformer':
        tokenizer_token = 'Ġ'

    num_logits = logits.shape[-2]

    assert num_logits - 2 == len(tokens), f"Assertion failed: num_logits - 2 ({num_logits - 2}) is not equal to len(tokens) ({len(tokens)})" # logits will include start and end tokens

    # Aggregate the logits for the first word
    left_logit_idx, right_logit_idx = 0, 2  # All final logits for first word will include logits for START and unmakred first subtoken
    while not tokens[right_logit_idx-1].startswith(tokenizer_token):
        right_logit_idx += 1

    processed_logits = torch.mean(logits[:, :right_logit_idx, :], dim=1, keepdim=True)

    # Second subtoken idx of second word
    second_word_idx = right_logit_idx
    left_logit_idx += right_logit_idx
    right_logit_idx += 1

    # Iterate through remaining subtokens
    for i in range(second_word_idx, len(tokens)):

        # If subtoken starts a new word,
        if tokens[i].startswith(tokenizer_token):

            # aggregate current window of logits
            logits_to_aggregate = logits[:, left_logit_idx:right_logit_idx, :]
            aggregated_logit = torch.mean(logits_to_aggregate, dim=1, keepdim=True)
            processed_logits = torch.cat([processed_logits, aggregated_logit], dim=1)

            # and set new window indicies
            left_logit_idx = right_logit_idx
            right_logit_idx += 1

        # If subtoken part of current word
        else:
            right_logit_idx += 1

    # Aggregate logits of final word
    logits_to_aggregate = logits[:, left_logit_idx:, :]
    aggregated_logit = torch.mean(logits_to_aggregate, dim=1, keepdim=True)
    processed_logits = torch.cat([processed_logits, aggregated_logit], dim=1)

    # Each word should have a unique logit after aggregation processing
    assert processed_logits.shape[-2] == word_sequence_length, f"Assertion failed: processed_logits.shape[-2] ({processed_logits.shape[-2]}) is not equal to len(tokens) ({word_sequence_length}: {tokens})"

    return processed_logits



def create_and_load_data(config):
    """
    create_and_load_data: Create a batch of training data for a set of clients. Load and Prepare train set

    Args:
        config: user-defined .ini config

    Returns:
        trainloaders: set of NERD training datasets
        valloaders: set of NERD training datasets
        testloader: the test set
        unique_tags: set of unique labels
    """


    # Parse config
    model_name = config['model']['model_name_or_path']
    data_dir = config['data']['data_dir']
    xml_dir = config['data']['xml_dir']

    num_clients = config.getint('fed','num_clients')

    max_seq_len = config.getint('model','max_seq_len')
    pad_token_label_id = config.getint('model','pad_token_label_id')
    seed = config.getint('model','seed')


    # Load partition XML training files for a number of clients 
    xml_paths = get_xml_files(xml_dir)
    partition_size, remainder = len(xml_paths) // num_clients, len(xml_paths) % num_clients
    lengths = [partition_size] * num_clients
    for i in range(remainder):
        lengths[i % len(lengths)] += 1
    datasets = random_split(xml_paths, lengths, torch.Generator().manual_seed(seed))

    tokenizer = LongformerTokenizerFast.from_pretrained(model_name)

    trainloaders = []
    valloaders = []
    unique_tags = get_labels(data_dir + 'labels.txt')


    # load the test set
    test_examples = read_examples_from_file(data_dir, 'test', max_seq_len=max_seq_len)
    testloader = NERDataset(convert_examples_to_features(test_examples,
                                              unique_tags,
                                              max_seq_length=max_seq_len,
                                              tokenizer=tokenizer,
                                              pad_token_label_id=pad_token_label_id))

    # Create Training and Test Set for Each Client
    for i, ds in enumerate(datasets):
        len_val = len(ds) // 10
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(seed))

        train_name = 'train-fed-{}'.format(i)
        val_name = 'dev-fed-{}'.format(i)

        _, train_labels = xml2bio(ds_train, output_file=data_dir + train_name + '.txt', preprocess=True)
        _, val_labels = xml2bio(ds_val, output_file=data_dir + val_name + '.txt', preprocess=True)

        train_examples = read_examples_from_file(data_dir, train_name)
        val_examples = read_examples_from_file(data_dir, val_name)

        train_features = convert_examples_to_features(train_examples,
                                                unique_tags,
                                                max_seq_length=max_seq_len,
                                                tokenizer=tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        val_features = convert_examples_to_features(val_examples,
                                                      unique_tags,
                                                      max_seq_length=max_seq_len,
                                                      tokenizer=tokenizer,
                                                      pad_token_label_id=pad_token_label_id)

        trainloaders.append(NERDataset(train_features))
        valloaders.append(NERDataset(val_features))

    return trainloaders, valloaders, testloader, unique_tags


