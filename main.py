import argparse
import glob
import sys
import tqdm
from distutils.util import strtobool
from dataset import *
from model import *
from utils import *


def get_config():
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
        required=False
    )

    parser.add_argument(
        "--backbone_name",
        nargs="?",
        type=str,
        required="--optimization_split_paths" in sys.argv
    )

    parser.add_argument(
        "--code_size",
        nargs="?",
        type=int,
        required="--optimization_split_paths" in sys.argv
    )

    parser.add_argument(
        "--overlap_loss_weight",
        nargs="?",
        type=float,
        required="--optimization_split_paths" in sys.argv
    )

    parser.add_argument(
        "--initial_temperature",
        nargs="?",
        type=float,
        required="--optimization_split_paths" in sys.argv
    )

    parser.add_argument(
        "--use_gated_fusion",
        nargs="?",
        type=lambda value: bool(strtobool(value)),
        required="--optimization_split_paths" in sys.argv
    )

    parser.add_argument(
        "--use_heuristic_negative",
        nargs="?",
        type=lambda value: bool(strtobool(value)),
        required="--optimization_split_paths" in sys.argv
    )

    parser.add_argument(
        "--backbone_activity",
        nargs="?",
        type=float,
        required="--optimization_split_paths" in sys.argv
    )

    parser.add_argument(
        "--learning_rate",
        nargs="?",
        type=float,
        required="--optimization_split_paths" in sys.argv
    )

    parser.add_argument(
        "--warmup_epoch_count",
        nargs="?",
        type=int,
        required="--optimization_split_paths" in sys.argv
    )

    parser.add_argument(
        "--total_epoch_count",
        nargs="?",
        type=int,
        required="--optimization_split_paths" in sys.argv
    )

    parser.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        required="--optimization_split_paths" in sys.argv
    )

    parser.add_argument(
        "--tcm_loss_weight",
        nargs="?",
        type=float,
        required="--optimization_split_paths" in sys.argv
    )

    parser.add_argument(
        "--itm_loss_weight",
        nargs="?",
        type=float,
        required="--optimization_split_paths" in sys.argv
    )

    config = vars(parser.parse_args())
    return config


def build(
        config,
        device
):
    if config["optimization_split_paths"] is not None:
        cir_model = CIRModel(
            config["backbone_name"],
            config["code_size"],
            config["overlap_loss_weight"],
            config["initial_temperature"],
            config["use_gated_fusion"],
            config["use_heuristic_negative"]
        )

        cir_model.to(device)
        backbone_decay_parameters = []
        backbone_other_parameters = []

        for module in [
            cir_model.image_encoder.backbone_encoder,
            cir_model.text_encoder.backbone_encoder
        ]:
            if config["backbone_activity"]:
                for name, parameter in module.named_parameters():
                    if "bias" not in name.lower() and "norm" not in name.lower():
                        backbone_decay_parameters.append(parameter)

                    else:
                        backbone_other_parameters.append(parameter)

            else:
                module.requires_grad_(False)

        head_decay_parameters = []
        head_other_parameters = []

        for module in [
            cir_model.image_encoder.code_projector,
            cir_model.text_encoder.code_projector,
            cir_model.fusion_module
        ]:
            for name, parameter in module.named_parameters():
                if "bias" not in name.lower() and "norm" not in name.lower():
                    head_decay_parameters.append(parameter)

                else:
                    head_other_parameters.append(parameter)

        head_other_parameters.extend(
            [
                cir_model.ritm_temperature,
                cir_model.titm_temperature,
                cir_model.icm_temperature,
                cir_model.tcm_temperature
            ]
        )

        cir_optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_decay_parameters,
                    "lr": config["backbone_activity"] * config["learning_rate"]
                },
                {
                    "params": backbone_other_parameters,
                    "lr": config["backbone_activity"] * config["learning_rate"],
                    "weight_decay": 0.0
                },
                {"params": head_decay_parameters},
                {
                    "params": head_other_parameters,
                    "weight_decay": 0.0
                }
            ],
            config["learning_rate"]
        )

        cir_scheduler = transformers.get_cosine_schedule_with_warmup(
            cir_optimizer,
            config["warmup_epoch_count"],
            config["total_epoch_count"]
        )

        cir_scaler = torch.cuda.amp.GradScaler()

        optimization_dataset = CIRDataset(
            os.path.join(
                root_path,
                config["image_base_path"]
            ),
            [
                os.path.join(
                    root_path,
                    path
                )
                for path in config["optimization_split_paths"]
            ]
        )

    else:
        cir_model = None
        cir_optimizer = None
        cir_scheduler = None
        cir_scaler = None
        optimization_dataset = None

    evaluation_dataset = CIRDataset(
        os.path.join(
            root_path,
            config["image_base_path"]
        ),
        [
            os.path.join(
                root_path,
                path
            )
            for path in config["evaluation_split_paths"]
        ]
    )

    return cir_model, cir_optimizer, cir_scheduler, cir_scaler, optimization_dataset, evaluation_dataset


def optimize(
        config,
        device,
        cir_model,
        cir_optimizer,
        cir_scheduler,
        cir_scaler,
        cir_loader
):
    for cir_batch in tqdm.tqdm(cir_loader):
        cir_batch.to(device)
        cir_optimizer.zero_grad(True)

        with torch.cuda.amp.autocast():
            ritm_loss, titm_loss, icm_loss, tcm_loss = cir_model(cir_batch)

            total_loss = torch.add(
                torch.add(
                    icm_loss,
                    tcm_loss,
                    alpha=config["tcm_loss_weight"]
                ),
                torch.add(
                    ritm_loss,
                    titm_loss
                ),
                alpha=config["itm_loss_weight"]
            )

        total_loss = cir_scaler.scale(total_loss)
        total_loss.backward()
        cir_scaler.step(cir_optimizer)
        cir_scaler.update()

    cir_scheduler.step()


def evaluate(
        config,
        device,
        cir_model,
        evaluation_dataset
):
    round_count, sample_count = config["evaluation_retrieval_scheme"]
    cir_performance = {}

    for _ in range(round_count):
        queries, labels, unique_images, image_groups = evaluation_dataset.get_retrieval_data(sample_count)

        unique_image_representations = cir_model.image_encoder.execute(
            torch.utils.data.DataLoader(
                unique_images,
                config["batch_size"],
                collate_fn=lambda images: images
            ),
            device
        )

        reference_image_indices = torch.tensor(
            [
                reference_image_index
                for reference_image_index, _ in queries
            ],
            dtype=torch.long,
            device=device
        )

        _, image_retrieval_result = torch.topk(
            torch.scatter(
                torch.matmul(
                    cir_model.fusion_module(
                        torch.index_select(
                            unique_image_representations,
                            0,
                            reference_image_indices
                        ),
                        cir_model.text_encoder.execute(
                            torch.utils.data.DataLoader(
                                [
                                    modification_text
                                    for _, modification_text in queries
                                ],
                                config["batch_size"],
                                collate_fn=lambda texts: texts
                            ),
                            device
                        )
                    ),
                    torch.t(unique_image_representations)
                ),
                1,
                torch.unsqueeze(
                    reference_image_indices,
                    1
                ),
                torch.unsqueeze(
                    torch.full(
                        reference_image_indices.shape,
                        float("-inf"),
                        dtype=torch.float,
                        device=device
                    ),
                    1
                )
            ),
            max(config["evaluation_metric_ks"])
        )

        group_retrieval_result = torch.gather(
            torch.broadcast_to(
                torch.unsqueeze(
                    torch.tensor(
                        image_groups,
                        dtype=torch.long,
                        device=device
                    ),
                    0
                ),
                [
                    image_retrieval_result.shape[0],
                    len(image_groups)
                ]
            ),
            1,
            image_retrieval_result
        )

        for k in config["evaluation_metric_ks"]:
            metric = "recall_at_{}".format(k)

            if metric not in cir_performance:
                cir_performance[metric] = []

            cir_performance[metric].append(
                torch.mean(
                    torch.any(
                        torch.eq(
                            group_retrieval_result[:, :k],
                            torch.unsqueeze(
                                torch.tensor(
                                    labels,
                                    dtype=torch.long,
                                    device=device
                                ),
                                1
                            )
                        ),
                        1
                    ).float()
                ).item()
            )

    for metric in cir_performance:
        cir_performance[metric] = sum(cir_performance[metric]) / len(cir_performance[metric])

    cir_performance["overall"] = sum(cir_performance.values()) / len(cir_performance)
    return cir_performance


def main(
        config,
        device
):
    cir_model, cir_optimizer, cir_scheduler, cir_scaler, optimization_dataset, evaluation_dataset = build(
        config,
        device
    )

    if cir_model is not None:
        if not glob.glob(outcome_path):
            os.mkdir(outcome_path)

        if glob.glob(checkpoint_path) and glob.glob(archive_path):
            checkpoint = torch.load(
                checkpoint_path,
                device
            )

            cir_model.load_state_dict(checkpoint["model_state"])
            cir_optimizer.load_state_dict(checkpoint["optimizer_state"])
            cir_scheduler.load_state_dict(checkpoint["scheduler_state"])
            cir_scaler.load_state_dict(checkpoint["scaler_state"])
            archive = load_file(archive_path)

        else:
            archive = {
                "config": config,
                "progress": [],
                "optimum": None
            }

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

            print(
                "epoch {}: {}.".format(
                    cir_scheduler.last_epoch,
                    json.dumps(cir_performance)
                ),
                end=" "
            )

            torch.save(
                {
                    "model_state": cir_model.state_dict(),
                    "optimizer_state": cir_optimizer.state_dict(),
                    "scheduler_state": cir_scheduler.state_dict(),
                    "scaler_state": cir_scaler.state_dict()
                },
                checkpoint_path
            )

            print(
                "checkpoint saved.",
                end=" "
            )

            archive["progress"].append(cir_performance)

            if archive["optimum"] is None or archive["optimum"]["overall"] < cir_performance["overall"]:
                torch.save(
                    cir_model,
                    release_path
                )

                print(
                    "release saved.",
                    end=" "
                )

                archive["optimum"] = cir_performance

            dump_file(
                archive,
                archive_path
            )

            print("archive saved.")

    else:
        cir_model = torch.load(
            release_path,
            device
        )

        cir_model.eval()

        with torch.no_grad():
            cir_performance = evaluate(
                config,
                device,
                cir_model,
                evaluation_dataset
            )

        setting = "retrieval on {} samples from {} for {} rounds".format(
            config["evaluation_retrieval_scheme"][1] if config["evaluation_retrieval_scheme"][1] > 0 else "all",
            ", ".join(config["evaluation_split_paths"]),
            config["evaluation_retrieval_scheme"][0]
        )

        print(
            "{}: {}.".format(
                setting,
                json.dumps(cir_performance)
            ),
            end=" "
        )

        archive = load_file(archive_path)
        archive[setting] = cir_performance

        dump_file(
            archive,
            archive_path
        )

        print("archive saved.")


if __name__ == "__main__":
    main(
        get_config(),
        torch.device("cuda")
    )
