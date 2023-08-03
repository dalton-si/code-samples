
import flwr as fl
import multiprocessing as mp
import numpy as np
from src.models.federated.flower_helpers import train, test, train_adam, create_model_components, get_params, set_params
from collections import OrderedDict # Will not need
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Some weights of the model checkpoint .* ")

class DEIDClient(fl.client.NumPyClient):

    def __init__(self, cid: int, trainloaders: list, valloaders: list, unique_tags, config, toy):
        self.cid = int(cid)
        self.config = config
        self.net = create_model_components(config, unique_tags)[-1]
        self.trainloader = trainloaders[self.cid]
        self.valloader = valloaders[self.cid]
        self.unique_tags = unique_tags
        self.num_epochs = self.config.getint('model', 'num_epochs')
        self.toy = toy


    def compute_metrics(self,pred):
        label_list = self.unique_tags
        predictions, labels = [i.tolist() for i in pred]

        pad_token_label_id = self.config.getint('model', 'pad_token_label_id')
        metric = load_metric("seqeval")

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != pad_token_label_id]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != pad_token_label_id]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"]
        }

    def get_parameters(self, config):
        return get_params(self.net)



    def fit(self, parameters, config):
        set_params(self.net, parameters)
        mp.set_start_method("spawn")
        manager = mp.Manager()
        return_dict = manager.dict()
        print("Training Started...")
        p = mp.Process(target=train, args=(self.cid, self.net, self.num_epochs, return_dict, self.config, self.trainloader, self.toy))
        p.start()
        p.join()
        try:
            p.close()
        except ValueError as e:
            print(f"Couldn't close the training process: {e}")


        new_parameters = return_dict["parameters"]
        data_size = return_dict["data_size"]
        accuracy = return_dict["accuracy"]
        print("Training Finished.")
        del (manager, return_dict, p)
        assert not np.array_equal(np.array(parameters),np.array(new_parameters))

        return new_parameters, data_size, {"accuracy": float(accuracy)}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)
        manager = mp.Manager()
        return_dict = manager.dict()
        p = mp.Process(target=test, args=(self.config, self.net, self.valloader, parameters, return_dict, self.toy))
        p.start()
        # Wait for it to end
        p.join()
        # Close it
        try:
            p.close()
        except ValueError as e:
            print(f"Coudln't close the evaluating process: {e}")
        # Get the return values
        loss = return_dict["loss"]
        accuracy = return_dict["accuracy"]
        data_size = return_dict["data_size"]
        # Del everything related to multiprocessing
        del (manager, return_dict, p)
        return float(loss), data_size, {"accuracy": float(accuracy)}
