from ray import tune
from ray.tune import CLIReporter
from main import *


def hpo(
        config,
        device
):
    cir_model, cir_optimizer, cir_scheduler, cir_scaler, optimization_dataset, evaluation_dataset = build(
        config,
        device
    )

    cir_loader = optimization_dataset.get_cir_loader(
        config["batch_size"],
        cir_model.image_encoder.backbone_extractor,
        cir_model.text_encoder.backbone_tokenizer
    )

    while cir_scheduler.last_epoch < config["total_epoch_count"]:
        cir_model.train()

        optimize(
            config,
            device,
            cir_model,
            cir_optimizer,
            cir_scheduler,
            cir_scaler,
            cir_loader
        )

        cir_model.eval()

        with torch.no_grad():
            cir_performance = evaluate(
                config,
                device,
                cir_model,
                evaluation_dataset
            )

        tune.report(overall=cir_performance["overall"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_base_path",
        nargs="?",
        type=str,
        required=True
    )

    parser.add_argument(
        "--evaluation_split_paths",
        nargs="+",
        type=str,
        required=True
    )

    parser.add_argument(
        "--evaluation_retrieval_scheme",
        nargs=2,
        type=int,
        required=True
    )

    parser.add_argument(
        "--evaluation_metric_ks",
        nargs="+",
        type=int,
        required=True
    )

    parser.add_argument(
        "--optimization_split_paths",
        nargs="+",
        type=str,
        required=True
    )

    parser.add_argument(
        "--backbone_name",
        nargs="?",
        type=str,
        required=True
    )

    args = parser.parse_args()

    result = tune.run(
        functools.partial(
            hpo,
            device=torch.device("cuda")
        ),
        config={
            "image_base_path": args.image_base_path,
            "evaluation_split_paths": args.evaluation_split_paths,
            "evaluation_retrieval_scheme": args.evaluation_retrieval_scheme,
            "evaluation_metric_ks": args.evaluation_metric_ks,
            "optimization_split_paths": args.optimization_split_paths,
            "backbone_name": args.backbone_name,
            "code_size": 512,
            "overlap_loss_weight": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
            "initial_temperature": tune.grid_search([0.0, 1.0, 2.0]),
            "use_gated_fusion": True,
            "use_heuristic_negative": True,
            "backbone_activity": tune.grid_search([1e-4, 1e-3, 1e-2]),
            "learning_rate": tune.grid_search([5e-5, 1e-4, 5e-4]),
            "warmup_epoch_count": 6,
            "total_epoch_count": 64,
            "batch_size": tune.grid_search([32, 64, 128]),
            "tcm_loss_weight": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
            "itm_loss_weight": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5])
        },
        resources_per_trial={
            "cpu": 4,
            "gpu": 1
        },
        progress_reporter=CLIReporter(
            [
                "training_iteration",
                "overall"
            ],
            [
                "overlap_loss_weight",
                "initial_temperature",
                "backbone_activity",
                "learning_rate",
                "batch_size",
                "tcm_loss_weight",
                "itm_loss_weight"
            ]
        )
    )

    best_trial = result.get_best_trial(
        "overall",
        "max",
        "all"
    )

    print("best trial id: {}".format(best_trial.trial_id))
    print("best trial config: {}".format(best_trial.config))
