from PIL import ImageDraw, ImageFont, ImageOps
from main import *


def infer(
        config,
        device
):
    cir_model = torch.load(
        release_path,
        device
    )

    cir_model.eval()

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

    round_count, sample_count = config["evaluation_retrieval_scheme"]
    queries, labels, unique_images, image_groups = evaluation_dataset.get_retrieval_data(sample_count)

    with torch.no_grad():
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
                        reference_image_indices.size(),
                        float("-inf"),
                        dtype=torch.float,
                        device=device
                    ),
                    1
                )
            ),
            max(config["evaluation_metric_ks"])
        )

        return queries, labels, unique_images, image_groups, image_retrieval_result


def draw_text(
        image,
        text,
        font
):
    x = 10
    y = 10
    width = image.width - 2 * x
    height = image.height - 2 * y
    lines = text.split("\n")
    true_lines = []

    for line in lines:
        if font.getsize(line)[0] <= width:
            true_lines.append(line)

        else:
            current_line = ""

            for word in line.split(" "):
                if font.getsize(current_line + word)[0] <= width:
                    current_line += word + " "

                else:
                    true_lines.append(current_line)
                    current_line = word + " "

            true_lines.append(current_line)

    y += height // 2
    line_height = font.getsize(true_lines[0])[1] * 1.5
    y_offset = - (len(true_lines) * line_height) / 2
    image_draw = ImageDraw.Draw(image)

    for line in true_lines:
        image_draw.text(
            (x, y + y_offset),
            line,
            "red",
            font
        )

        y_offset += line_height


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

    args = parser.parse_args()

    queries, labels, unique_images, image_groups, image_retrieval_result = infer(
        {
            "image_base_path": args.image_base_path,
            "evaluation_split_paths": args.evaluation_split_paths,
            "evaluation_retrieval_scheme": [1, 0],
            "evaluation_metric_ks": [10],
            "batch_size": 64
        },
        torch.device("cuda")
    )

    font = ImageFont.truetype("font.ttf", 16)

    for index, retrieval_image_indices in enumerate(image_retrieval_result.tolist()):
        reference_image_index, modification_text = queries[index]
        reference_image = unique_images[reference_image_index]

        modification_image = Image.new(
            "RGB",
            (128, 128),
            "white"
        )

        draw_text(
            modification_image,
            modification_text,
            font
        )

        images = [reference_image, modification_image]
        rank = 0

        for retrieval_index, retrieval_image_index in enumerate(retrieval_image_indices):
            retrieval_image = unique_images[retrieval_image_index]

            if labels[index] == image_groups[retrieval_image_index]:
                retrieval_image = ImageOps.expand(
                    retrieval_image,
                    4,
                    "red"
                )

                if rank == 0:
                    rank = retrieval_index + 1

            images.append(retrieval_image)

        joint_image = Image.new(
            "RGB",
            (len(images) * 128, 128),
            "white"
        )

        for image_index, image in enumerate(images):
            joint_image.paste(
                image.resize((128, 128)),
                (image_index * 128, 0)
            )

        folder_path = os.path.join(
            outcome_path,
            "{}".format(rank)
        )

        if not glob.glob(folder_path):
            os.mkdir(folder_path)

        joint_image.save(
            os.path.join(
                folder_path,
                os.path.basename(reference_image.path)
            )
        )
