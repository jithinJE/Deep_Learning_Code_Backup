# Handles running selected model against selected date ranges
import dataload
import pprint
import datetime
from keras.models import load_model
import os
from train import plot_results_multiple

def main():
    #model = load_model('test_models/stock_multb_nosmooth_step4_1525242340/seq_len_50/model-1.h5')
    model = load_model('D:\Source\Repos\StockPrediction\stock_single_nosmooth_1525405252\seq_len_50/model-1.h5')
    seq_len = 50
    predict_len = 7

    #date_ranges = [(datetime.date(2016,11,1),datetime.date(2017,4,3)),(datetime.date(2002,3,18),datetime.date(2002,8,14)),(datetime.date(2015,3,8),datetime.date(2015,8,5))]
    date_ranges = [(datetime.date(2015,1,1),datetime.date(2015,6,1)),
		    (datetime.date(2016,1,1),datetime.date(2016,6,1)),
		    (datetime.date(2017,1,1),datetime.date(2017,6,1))]

    test_data = [dataload.load_data('daily_spx.csv', seq_len, normalise_window=True, smoothing=False, date_range=date_range, train=False) for date_range in date_ranges]

    predictions = [dataload.predict_sequences_multiple(model, test[0], seq_len, predict_len) for test in test_data]
    scores = [model.evaluate(test[0], test[1], verbose=0) for test in test_data]

    for prediction_index in range(len(predictions)):
        for sequence_index in range(len(predictions[prediction_index])):
            predictions[prediction_index][sequence_index] = dataload.denormalize_sequence(test_data[prediction_index][2][sequence_index*7], predictions[prediction_index][sequence_index])

    for test_data_index in range(len(test_data)):
        for y_index in range(len(test_data[test_data_index][1])):
            test_data[test_data_index][1][y_index] = dataload.denormalize_point(test_data[test_data_index][2][y_index], test_data[test_data_index][1][y_index])

    model_plot = [(predictions[0], '2015'), (predictions[1], '2016'), (predictions[2], '2017')]
    #model_plot = [(predictions[0], 'Bullish'), (predictions[1], 'Bearish'), (predictions[2], 'Neutral')]
    results_fname = 'test_singleb_nosmooth_byyear_{}'.format(int(datetime.datetime.now().timestamp()))
    os.makedirs(results_fname)
    plot_results_multiple(model_plot, [t[1] for t in test_data], predict_len, fig_path = results_fname + '/plots.pdf')
    with open(results_fname + "/score.txt", "w") as fout:
        for score in scores:
            pprint.pprint(score, fout)

if __name__ == '__main__':
    main()
