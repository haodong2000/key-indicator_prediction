global_params = {
    "data_path": "./data/NSC_Si_Content_Timedelay_Data_CN.xlsx",
    "model_path": "./models",
    "log_path": "./logs",
    "time_zone": "Asia/Singapore",
    "window_size": 500,
    "num_of_step": 1,
    "select_dim": [0] + [52 + i for i in range(5)],
    "split_ratio": 0.7,
    "epochs": 100,
    "learning_rate": 0.001,
    "decay_rate": 1e1 * 0.001/100, # learning_rate/epochs
    "optimizer": "Nadam",
    "monitor": "val_loss",
    "patience": 50,
    "verbose": 1,
    "save_best_only": True,
    "shuffle": True,
    "shuffle_data_split": True,
    "batch_size": 64,
    "output_dim": [0] + [52 + i for i in range(5)],
    "expert_know": ["(t - " + str(i) + ")" for i in range(1, 6)]
}

prediction = global_params.copy()
prediction.update({
    "num_of_step": 1,
    "delay": 0,
    "simple_lstm": {
        "window_size": 50,
        "patience": 50,
        "epochs": 200,
        "learning_rate": 0.00004,
        "decay_rate": 0.00004/200,
        "batch_size": 64
    },
    "cnn_rnn": {
        "window_size": 50,
        "num_of_conv": 2, 
        "append_rnn": True,
        "patience": 50,
        "epochs": 200,
        "learning_rate": 0.00004,
        "batch_size": 64,
        "decay_rate": 0.00004/200
    },
    "resnet_rnn": {
        "window_size": 50,
        "block_num": 2,
        "append_rnn": True,
        "patience": 50,
        "epochs": 200,
        "learning_rate": 0.00004,
        "batch_size": 64,
        "decay_rate": 0.00004/200
    },
    "efficientnetv2_rnn": {
        "window_size": 50,
        "weights": None,
        "include_top": False,
        "pooling": "avg",
        "append_rnn": True,
        "patience": 50,
        "epochs": 200,
        "learning_rate": 0.00004,
        "batch_size": 64,
        "decay_rate": 0.00004/200
    },
    "transformer": {
        "feature_size": 512, 
        "num_of_var": 6,
        "select_dim": 500, 
        "num_layers": 1, 
        "num_head": 16, 
        "dropout": 0.1,
        "epochs": 200,
        "learning_rate": 0.00005,
        "batch_size": 64,
        "decay_rate": 0.00005/200
    },
    "input_dim": global_params["output_dim"],
    "continual_learning": {
        "seq_length": 50,
        "seq_length_long": 500,
        "output_dim_1": 32,
        "output_dim_2": 32,
        "input_dim_new": 6,
        "output_dim": 6,
        "feature_size": 256,
        "num_heads": 1, 
        "dropout": 0.1,
        "num_layers": 2,
        "epochs": 200,
        "learning_rate": 0.0002,
        "batch_size": 64,
        "decay_rate": 0.0002/1000
    },
})

soft = global_params.copy()
soft.update({
    "batch_size": 64,
    "epochs": 200,
    "learning_rate": 0.0001,
    "decay_rate": 0.0001/200,
})

model_configs = [
    ["simple_lstm", "simple_lstm"], 
    ["cnn_rnn_no_batchnorm", "cnn_rnn"], 
    ["resnet_rnn_layernorm", "resnet_rnn"], 
    ["efficientnetv2_rnn", "efficientnetv2_rnn"],
    ["transformer_1", "transformer"],
    ["continual_learning_1", "continual_learning"]
]
