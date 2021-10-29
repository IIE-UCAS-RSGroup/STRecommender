# -*- coding: utf-8 -*-
import argparse
from config.configurator import Config
from utils.utils import *
from utils.data_preprocessing import *
from evaluator.metrics import precisionN, recallN, mapN, ndcgN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UserBasedCF', help='name of models')
    parser.add_argument('--dataset', type=str, default='Foursquare', help='name of datasets')
    parser.add_argument('--rec_num', type=int, default=20, help='number of recommendation results')
    parser.add_argument('--top_n_list', type=list, default=[1, 3, 5], help='scope of evaluation')
    parser.add_argument('--config_file', type=str, default='./properties/overall.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file = args.config_file.strip().split(' ') if args.config_file else None

    config = Config(config_file_list=config_file)

    print(config.final_config_dict)

    data_dir = "./dataset/Foursquare/"

    size_file = data_dir + "Foursquare_data_size.txt"
    train_file = data_dir + "Foursquare_train.txt"
    test_file = data_dir + "Foursquare_test.txt"
    poi_file = data_dir + "Foursquare_poi_coos.txt"

    user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num = int(user_num), int(poi_num)

    training_matrix = read_training_data(train_file, user_num, poi_num, config)
    ground_truth = read_ground_truth(test_file)
    # poi_coos = read_poi_coos(poi_file)

    model = get_model(config['model'])(config)

    model.compute_rec_scores(training_matrix, config)

    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    result_out = open("./results/" + config['model'] + '_' + config['dataset'] + '_' + str(config['rec_num']) + ".txt", 'w')

    preN = precisionN(config)
    recN = recallN(config)
    mapN = mapN(config)
    ndcgN = ndcgN(config)

    print("Start predicting...")
    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            I_scores = normalize([model.predict(uid, lid)  for lid in all_lids])

            overall_scores = np.array(I_scores)

            predicted = list(reversed(overall_scores.argsort()))[:config['rec_num']]
            actual = ground_truth[uid]

            result_out.write('\t'.join([str(cnt), str(uid), ','.join([str(lid) for lid in predicted])]) + '\n')

            preN.compute(actual, predicted)
            recN.compute(actual, predicted)
            mapN.compute(actual, predicted)
            ndcgN.compute(actual, predicted)

    for i in range(len(config['top_n_list'])):
        print("pre@%d" % config['top_n_list'][i], preN.result()[i], "rec@%d" % config['top_n_list'][i],
              recN.result()[i], "map@%d" % config['top_n_list'][i],
              mapN.result()[i], "ndcg@%d" % config['top_n_list'][i], ndcgN.result()[i])
    print("Task Finished!")

