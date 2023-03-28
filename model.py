import torch
import transformers


class ImageEncoder(torch.nn.Module):
    def __init__(
            self,
            backbone_extractor,
            backbone_encoder,
            code_size
    ):
        super().__init__()
        self.backbone_extractor = backbone_extractor
        self.backbone_encoder = backbone_encoder

        self.code_projector = torch.nn.Linear(
            self.backbone_encoder.config.hidden_size,
            code_size
        )

    def forward(
            self,
            image_features
    ):
        image_codes = torch.nn.functional.normalize(
            self.code_projector(
                self.backbone_encoder(
                    image_features
                ).pooler_output
            )
        )

        return image_codes

    def execute(
            self,
            image_loader,
            device
    ):
        image_codes = []

        for image_batch in image_loader:
            image_extraction = self.backbone_extractor(
                image_batch,
                return_tensors="pt"
            )

            image_features = image_extraction["pixel_values"]
            image_features = image_features.to(device)

            image_codes.append(
                self(
                    image_features
                )
            )

        image_codes = torch.cat(image_codes)
        return image_codes


class TextEncoder(torch.nn.Module):
    def __init__(
            self,
            backbone_tokenizer,
            backbone_encoder,
            code_size
    ):
        super().__init__()
        self.backbone_tokenizer = backbone_tokenizer
        self.backbone_encoder = backbone_encoder

        self.code_projector = torch.nn.Linear(
            self.backbone_encoder.config.hidden_size,
            code_size
        )

    def forward(
            self,
            text_tokens,
            text_mask
    ):
        text_codes = torch.nn.functional.normalize(
            self.code_projector(
                self.backbone_encoder(
                    text_tokens,
                    text_mask
                ).pooler_output
            )
        )

        return text_codes

    def execute(
            self,
            text_loader,
            device
    ):
        text_codes = []

        for text_batch in text_loader:
            text_tokenization = self.backbone_tokenizer(
                text_batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            text_tokens = text_tokenization["input_ids"]
            text_tokens = text_tokens.to(device)
            text_mask = text_tokenization["attention_mask"]
            text_mask = text_mask.to(device)

            text_codes.append(
                self(
                    text_tokens,
                    text_mask
                )
            )

        text_codes = torch.cat(text_codes)
        return text_codes


class FusionModule(torch.nn.Module):
    def __init__(
            self,
            use_gated_fusion,
            code_size
    ):
        super().__init__()
        self.use_gated_fusion = use_gated_fusion

        if self.use_gated_fusion:
            self.gate_layer = torch.nn.Linear(
                code_size * 4,
                code_size
            )

            self.transform_layer = torch.nn.Linear(
                code_size * 4,
                code_size
            )

    def forward(
            self,
            subject_codes,
            object_codes
    ):
        if self.use_gated_fusion:
            joint_codes = torch.cat(
                [
                    subject_codes,
                    object_codes,
                    torch.mul(
                        subject_codes,
                        object_codes
                    ),
                    torch.sub(
                        subject_codes,
                        object_codes
                    )
                ],
                -1
            )

            fusion_gate = torch.nn.functional.sigmoid(
                self.gate_layer(
                    joint_codes
                )
            )

            fused_codes = torch.nn.functional.normalize(
                torch.add(
                    torch.mul(
                        torch.nn.functional.gelu(
                            self.transform_layer(
                                joint_codes
                            )
                        ),
                        fusion_gate
                    ),
                    torch.mul(
                        subject_codes,
                        torch.sub(
                            torch.ones_like(fusion_gate),
                            fusion_gate
                        )
                    )
                ),
                dim=-1
            )

        else:
            fused_codes = torch.nn.functional.normalize(
                torch.add(
                    subject_codes,
                    object_codes
                ),
                dim=-1
            )

        return fused_codes


class ContrastiveLoss(torch.nn.Module):
    def __init__(
            self,
            use_heuristic_negative,
            overlap_loss_weight
    ):
        super().__init__()
        self.use_heuristic_negative = use_heuristic_negative
        self.overlap_loss_weight = overlap_loss_weight

    def get_loss(
            self,
            logits,
            labels,
            is_symmetric
    ):
        if is_symmetric:
            loss = torch.add(
                torch.nn.functional.cross_entropy(
                    logits,
                    labels
                ),
                torch.nn.functional.cross_entropy(
                    torch.t(logits),
                    labels
                )
            )

        else:
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels
            )

        return loss

    def forward(
            self,
            subject_codes,
            object_codes,
            temperature
    ):
        logits = torch.mul(
            torch.matmul(
                subject_codes,
                torch.transpose(
                    object_codes,
                    -2,
                    -1
                )
            ),
            torch.exp(temperature)
        )

        labels = torch.arange(
            logits.shape[0],
            dtype=torch.long,
            device=logits.device
        )

        if logits.dim() == 2:
            loss = self.get_loss(
                logits,
                labels,
                True
            )

        else:
            if self.use_heuristic_negative:
                loss = torch.add(
                    torch.add(
                        self.get_loss(
                            torch.diagonal(
                                logits,
                                dim2=2
                            ),
                            labels,
                            True
                        ),
                        self.get_loss(
                            torch.diagonal(logits),
                            labels,
                            True
                        ),
                        alpha=self.overlap_loss_weight
                    ),
                    self.get_loss(
                        torch.diagonal(
                            logits,
                            dim1=1,
                            dim2=2
                        ),
                        labels,
                        True
                    ),
                    alpha=self.overlap_loss_weight
                )

            else:
                positive_logits = torch.unsqueeze(
                    torch.diagonal(
                        torch.diagonal(
                            logits
                        )
                    ),
                    1
                )

                negative_logits = torch.diagonal_scatter(
                    logits,
                    torch.diagonal_scatter(
                        torch.diagonal(logits),
                        torch.full(
                            [logits.shape[0]],
                            float("-inf"),
                            dtype=torch.float,
                            device=logits.device
                        )
                    )
                )

                loss = torch.add(
                    torch.add(
                        self.get_loss(
                            torch.cat(
                                [
                                    positive_logits,
                                    torch.topk(
                                        torch.reshape(
                                            negative_logits,
                                            [logits.shape[0], -1]
                                        ),
                                        logits.shape[0] - 1
                                    )[0]
                                ],
                                1
                            ),
                            torch.zeros_like(labels),
                            False
                        ),
                        self.get_loss(
                            torch.cat(
                                [
                                    positive_logits,
                                    torch.topk(
                                        torch.reshape(
                                            torch.transpose(
                                                negative_logits,
                                                0,
                                                1
                                            ),
                                            [logits.shape[0], -1]
                                        ),
                                        logits.shape[0] - 1
                                    )[0]
                                ],
                                1
                            ),
                            torch.zeros_like(labels),
                            False
                        )
                    ),
                    self.get_loss(
                        torch.cat(
                            [
                                positive_logits,
                                torch.topk(
                                    torch.reshape(
                                        torch.transpose(
                                            negative_logits,
                                            0,
                                            2
                                        ),
                                        [logits.shape[0], -1]
                                    ),
                                    logits.shape[0] - 1
                                )[0]
                            ],
                            1
                        ),
                        torch.zeros_like(labels),
                        False
                    )
                )

        return loss


class CIRModel(torch.nn.Module):
    def __init__(
            self,
            backbone_name,
            code_size,
            overlap_loss_weight,
            initial_temperature,
            use_gated_fusion,
            use_heuristic_negative
    ):
        super().__init__()
        backbone_processor = transformers.CLIPProcessor.from_pretrained(backbone_name)
        backbone_model = transformers.CLIPModel.from_pretrained(backbone_name)
        backbone_model.gradient_checkpointing_enable()

        self.image_encoder = ImageEncoder(
            backbone_processor.feature_extractor,
            backbone_model.vision_model,
            code_size
        )

        self.text_encoder = TextEncoder(
            backbone_processor.tokenizer,
            backbone_model.text_model,
            code_size
        )

        self.fusion_module = FusionModule(
            use_gated_fusion,
            code_size
        )

        self.contrastive_loss = ContrastiveLoss(
            use_heuristic_negative,
            overlap_loss_weight
        )

        self.ritm_temperature = torch.nn.Parameter(
            torch.tensor(
                initial_temperature,
                dtype=torch.float
            )
        )

        self.titm_temperature = torch.nn.Parameter(
            torch.tensor(
                initial_temperature,
                dtype=torch.float
            )
        )

        self.icm_temperature = torch.nn.Parameter(
            torch.tensor(
                initial_temperature,
                dtype=torch.float
            )
        )

        self.tcm_temperature = torch.nn.Parameter(
            torch.tensor(
                initial_temperature,
                dtype=torch.float
            )
        )

    def forward(
            self,
            cir_batch
    ):
        reference_image_codes = self.image_encoder(cir_batch.reference_image_features)

        reference_text_codes = self.text_encoder(
            cir_batch.reference_text_tokens,
            cir_batch.reference_text_mask
        )

        ritm_loss = self.contrastive_loss(
            reference_image_codes,
            reference_text_codes,
            self.ritm_temperature
        )

        target_image_codes = self.image_encoder(cir_batch.target_image_features)

        target_text_codes = self.text_encoder(
            cir_batch.target_text_tokens,
            cir_batch.target_text_mask
        )

        titm_loss = self.contrastive_loss(
            target_image_codes,
            target_text_codes,
            self.titm_temperature
        )

        modification_text_codes = self.text_encoder(
            cir_batch.modification_text_tokens,
            cir_batch.modification_text_mask
        )

        icm_loss = self.contrastive_loss(
            self.fusion_module(
                torch.broadcast_to(
                    torch.unsqueeze(
                        reference_image_codes,
                        1
                    ),
                    [
                        reference_image_codes.shape[0],
                        modification_text_codes.shape[0],
                        reference_image_codes.shape[1]
                    ]
                ),
                torch.broadcast_to(
                    torch.unsqueeze(
                        modification_text_codes,
                        0
                    ),
                    [
                        reference_image_codes.shape[0],
                        modification_text_codes.shape[0],
                        modification_text_codes.shape[1]
                    ]
                )
            ),
            torch.broadcast_to(
                torch.unsqueeze(
                    target_image_codes,
                    0
                ),
                [
                    reference_image_codes.shape[0],
                    target_image_codes.shape[0],
                    target_image_codes.shape[1]
                ]
            ),
            self.icm_temperature
        )

        tcm_loss = self.contrastive_loss(
            self.fusion_module(
                torch.broadcast_to(
                    torch.unsqueeze(
                        reference_text_codes,
                        1
                    ),
                    [
                        reference_text_codes.shape[0],
                        modification_text_codes.shape[0],
                        reference_text_codes.shape[1]
                    ]
                ),
                torch.broadcast_to(
                    torch.unsqueeze(
                        modification_text_codes,
                        0
                    ),
                    [
                        reference_text_codes.shape[0],
                        modification_text_codes.shape[0],
                        modification_text_codes.shape[1]
                    ]
                )
            ),
            torch.broadcast_to(
                torch.unsqueeze(
                    target_text_codes,
                    0
                ),
                [
                    reference_text_codes.shape[0],
                    target_text_codes.shape[0],
                    target_text_codes.shape[1]
                ]
            ),
            self.tcm_temperature
        )

        return ritm_loss, titm_loss, icm_loss, tcm_loss
