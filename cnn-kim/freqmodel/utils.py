from sklearn.utils import shuffle

import pickle


def read_TREC(dataset):
    data = {}

    def read(dataset, mode):
        x, y = [], []
        with open("data_readability/eng_".format(dataset) + mode + ".txt", "r", encoding="utf-8") as f:
        #with open("data/zho_" + mode + ".txt", "r", encoding="utf-8") as f:
        #with open("data/TREC/TREC_" + mode + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                y.append(line.split()[0].split(":")[0])
                x.append(line.split()[1:])

        x, y = shuffle(x, y)

        if mode == "train":
            dev_idx = len(x) // 10
            data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
            data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
        else:
            data["test_x"], data["test_y"] = x, y

    read(dataset, "train")
    read(dataset, "test")

    return data


def read_MR():
    data = {}
    x, y = [], []

    with open("data/MR/rt-polarity.pos", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("data/MR/rt-polarity.neg", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

    return data


def save_model(model, params):
#    path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    path = "saved_models/{0}_{1}_{2}.pkl".format(params['DATASET'],params['MODEL'],params['EPOCH'])
    pickle.dump(model, open(path, "wb"))
    print("A model is saved successfully as {0}!".format(path))


def load_model(params):
#    path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    path = "saved_models/{0}_{1}_{2}.pkl".format(params['DATASET'],params['MODEL'],params['EPOCH'])

    try:
        model = pickle.load(open(path, "rb"))
        print("Model in {0} loaded successfully!".format(path))

        return model
    except:
        print("No available model such as {0}.".format(path))
        exit()
