from learn_materials import utils
import pathlib

torch_datasets = utils.DATA_PATH / "materials_project" / "torch_datasets"

args_dict = {
    "mode": "train",
    "wandb": True,
    "wandb_entity": None,
    "wandb_project": None,
    "dryrun": True,
    "use_supergraphs": True,
    "model_type": "simple",
    "target_props": ["Fermi energy"],
    "loss": "mse",
    "epochs": 1000,
    "lr": 1e-3,
    "minimum_lr": 1e-6,
    "batch_size": 128,
    "max_batch": 128,
    "site_props": "hubbard",
    "supercell_size": 8,
    "max_distance": 6,
    "perov": False,
    ####
    "supergraphs_train_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_train.pt",
    "supergraphs_validation_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_valid.pt",
    "supergraphs_sample_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_sample.pt",
    #
    "magmoms_supergraphs_train_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_magmoms_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_train.pt",
    "magmoms_supergraphs_validation_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_magmoms_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_valid.pt",
    "magmoms_supergraphs_sample_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_magmoms_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_test.pt",
    #
    "insulators_supergraphs_train_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_insulators_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_train.pt",
    "insulators_supergraphs_validation_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_insulators_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_valid.pt",
    "insulators_supergraphs_sample_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_insulators_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_sample.pt",
    ####
    "supergraphs_hubbard_train_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_hubbard_train.pt",
    "supergraphs_hubbard_validation_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_hubbard_valid.pt",
    "supergraphs_hubbard_sample_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_hubbard_sample.pt",
    #
    "insulators_supergraphs_hubbard_train_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_insulators_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_hubbard_train.pt",
    "insulators_supergraphs_hubbard_validation_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_insulators_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_hubbard_valid.pt",
    "insulators_supergraphs_hubbard_sample_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_insulators_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_hubbard_sample.pt",
    ###
    "supergraphs_period_train_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_period_train.pt",
    "supergraphs_period_validation_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_period_valid.pt",
    "supergraphs_period_sample_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_period_sample.pt",
    #
    "insulators_period_supergraphs_train_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_insulators_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_period_train.pt",
    "insulators_period_supergraphs_validation_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_insulators_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_period_valid.pt",
    "insulators_period_supergraphs_sample_dataset": torch_datasets
    / "supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_insulators_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_period_sample.pt",
    ###
    "graphs_train_dataset": torch_datasets
    / "graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_train.pt",
    "graphs_validation_dataset": torch_datasets
    / "graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_valid.pt",
    "graphs_sample_dataset": torch_datasets
    / "graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_sample.pt",
    #
    "insulators_graphs_train_dataset": torch_datasets
    / "graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_insulators_graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_train.pt",
    "insulators_graphs_validation_dataset": torch_datasets
    / "gaphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_insulators_graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_valid.pt",
    "insulators_graphs_sample_dataset": torch_datasets
    / "graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_insulators_graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_sample.pt",
    ###
    "run_dataset": torch_datasets
    / "graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs"
    / "matproj_graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_test.pt",
    ###
    "supergraphs_train_dataset_0.5": torch_datasets / "supergraphs_subsample" / "train_data_0.5.pt",
    "supergraphs_train_dataset_0.25": torch_datasets / "supergraphs_subsample" / "train_data_0.25.pt",
    "supergraphs_train_dataset_0.125": torch_datasets / "supergraphs_subsample" / "train_data_0.125.pt",
    ###
    "perov_supergraphs_train": utils.DATA_PATH
    / "perov5"
    / "torch_datasets"
    / "train_supergraphs_all_cut=5.0_tol=0.5.pt",
    "perov_supergraphs_validation": utils.DATA_PATH
    / "perov5"
    / "torch_datasets"
    / "val_supergraphs_all_cut=5.0_tol=0.5.pt",
    "perov_supergraphs_test": utils.DATA_PATH / "perov5" / "torch_datasets" / "test_supergraphs_all_cut=5.0_tol=0.5.pt",
    "perov_graphs_test": utils.DATA_PATH / "perov5" / "torch_datasets" / "test_graphs_all_cut=5.0_tol=0.5.pt",
    ###
    "model_saves_dir": utils.RESULTS_PATH / "model_saves",
    "training_curves_dir": utils.RESULTS_PATH / "training_curves",
    "material_props": ["Mean atomic number"],
    "cuda": True,
    "seed": 72,
    "save_model": True,
    "patience": 25,
    "lr_cycle": 0,
    "factor": 0.5,
    "weight_decay": 1e-15,
    "schedule": True,
    "loss_output_len": 1,
    "class_threshold": 1e-5,
    "num_workers": 0,
    "saved_model": utils.RESULTS_PATH / "model_saves" / "megnet" / "metal_error",
    "lattices_data": [
        "ptriclinic",
        "pmonoclinic",
        "cmonoclinic",
        "porthorhombic",
        "corthorhombic",
        "borthorhombic",
        "forthorhombic",
        "ptetragonal",
        "btetragonal",
        "prhombohedral",
        "phexagonal",
        "pcubic",
        "bcubic",
        "fcubic",
    ],
}

help_dict = {
    "wandb": "Whether to save results to wandb serve",
    "wandb_entity": "Wandb account or team to which project belongs",
    "test": "triggers the test mode (1 epoch and stop)",
    "graphs": "Use dataset with graphs",
    "angles": "Use dataset with graphs and angle (dimenet)",
    "train_dataset": "path to the train dataset",
    "validation_dataset": "path to the validation dataset",
    "model_saves_dir": "path to the directory where to save models",
    "training_curves_dir": "path to the directory where to save training curves",
    "target_props": "list of targets used during training",
    "loss": "Set the loss function 'mse' or 'mae'",
    "epochs": "maximum number of epochs to train",
    "lr": "learning rate of the optimizer",
    "batch_size": "batch size used during training and validation",
    "material_props": "list of properties to use with the mlp",
    "missing_entries": "value used for missing entries for the mlp properties",
    "missing_uncert": "value used for missing entries for the mlp properties",
    "cuda": "Use the GPU",
    "seed": "Random seed",
    "save_model": "Will save models",
    "save_curve": "Will save training curves",
    "schedule": "Turns on the learning rate scheduler `reduce_on_plateau`",
    "patience": "Patience of the learning rate scheduler `reduce_on_plateau`",
    "factor": "Diminution factor of the learning rate scheduler `reduce_on_plateau`",
    "num_workers": "controls the number of CPU processes that load the data on the GPU",
    "mode": "how to run the mode. can be 'train', 'evaluate', 'inference'",
}


def create_full_hyperparams(ext_args_dict, ext_help_dict, argv):
    args_dict.update(ext_args_dict)
    help_dict.update(ext_help_dict)
    hyperparams = utils.read_args(args_dict, help_dict, argv=argv)
    if hyperparams.loss in ["gaussian", "quantile", "two_part"]:
        hyperparams.loss_output_len *= 2
    return hyperparams
